import sys
import time
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, '../../Utilities/')
from plotting import newfig, savefig

np.random.seed(1234)
tf.set_random_seed(1234)


class NLSSolver:
    def __init__(self, x_init, u_init, v_init, t_bnd, collocation_pts, arch, domain_min, domain_max):
        self.domain_min = domain_min
        self.domain_max = domain_max

        init_coords = np.concatenate((x_init, np.zeros_like(x_init)), axis=1)
        left_bnd = np.concatenate((np.full_like(t_bnd, domain_min[0]), t_bnd), axis=1)
        right_bnd = np.concatenate((np.full_like(t_bnd, domain_max[0]), t_bnd), axis=1)

        self.x0, self.t0 = init_coords[:, 0:1], init_coords[:, 1:2]
        self.x_left, self.t_left = left_bnd[:, 0:1], left_bnd[:, 1:2]
        self.x_right, self.t_right = right_bnd[:, 0:1], right_bnd[:, 1:2]
        self.x_col, self.t_col = collocation_pts[:, 0:1], collocation_pts[:, 1:2]
        self.u0 = u_init
        self.v0 = v_init
        self.arch = arch

        self.W_list, self.b_list = self._init_network_params(arch)

        self._build_graph()
        self._configure_optimizers()

        session_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.session = tf.Session(config=session_cfg)
        self.session.run(tf.global_variables_initializer())

    @staticmethod
    def _glorot_variable(fan_in, fan_out):
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return tf.Variable(tf.truncated_normal([fan_in, fan_out], stddev=scale), dtype=tf.float32)

    def _init_network_params(self, arch):
        weights, biases = [], []
        for idx in range(len(arch) - 1):
            weights.append(self._glorot_variable(arch[idx], arch[idx + 1]))
            biases.append(tf.Variable(tf.zeros([1, arch[idx + 1]], dtype=tf.float32), dtype=tf.float32))
        return weights, biases

    def _forward_pass(self, inputs):
        hidden = 2.0 * (inputs - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        for w_mat, b_vec in zip(self.W_list[:-1], self.b_list[:-1]):
            hidden = tf.tanh(tf.matmul(hidden, w_mat) + b_vec)
        return tf.matmul(hidden, self.W_list[-1]) + self.b_list[-1]

    def _evaluate_field(self, x_tf, t_tf):
        uv_pair = self._forward_pass(tf.concat([x_tf, t_tf], axis=1))
        u_real = uv_pair[:, 0:1]
        v_imag = uv_pair[:, 1:2]
        u_x = tf.gradients(u_real, x_tf)[0]
        v_x = tf.gradients(v_imag, x_tf)[0]
        return u_real, v_imag, u_x, v_x

    def _evaluate_pde_residual(self, x_tf, t_tf):
        u_real, v_imag, u_x, v_x = self._evaluate_field(x_tf, t_tf)
        u_t = tf.gradients(u_real, t_tf)[0]
        u_xx = tf.gradients(u_x, x_tf)[0]
        v_t = tf.gradients(v_imag, t_tf)[0]
        v_xx = tf.gradients(v_x, x_tf)[0]

        mod_sq = u_real ** 2 + v_imag ** 2
        res_u = u_t + 0.5 * v_xx + mod_sq * v_imag
        res_v = v_t - 0.5 * u_xx - mod_sq * u_real
        return res_u, res_v

    def _build_graph(self):
        self.ph_x0 = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.ph_t0 = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.ph_u0 = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.ph_v0 = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])

        self.ph_x_left = tf.placeholder(tf.float32, shape=[None, self.x_left.shape[1]])
        self.ph_t_left = tf.placeholder(tf.float32, shape=[None, self.t_left.shape[1]])
        self.ph_x_right = tf.placeholder(tf.float32, shape=[None, self.x_right.shape[1]])
        self.ph_t_right = tf.placeholder(tf.float32, shape=[None, self.t_right.shape[1]])

        self.ph_x_col = tf.placeholder(tf.float32, shape=[None, self.x_col.shape[1]])
        self.ph_t_col = tf.placeholder(tf.float32, shape=[None, self.t_col.shape[1]])

        self.u0_est, self.v0_est, _, _ = self._evaluate_field(self.ph_x0, self.ph_t0)
        u_l, v_l, ux_l, vx_l = self._evaluate_field(self.ph_x_left, self.ph_t_left)
        u_r, v_r, ux_r, vx_r = self._evaluate_field(self.ph_x_right, self.ph_t_right)
        self.fu_est, self.fv_est = self._evaluate_pde_residual(self.ph_x_col, self.ph_t_col)

        loss_terms = [
            tf.reduce_mean(tf.square(self.ph_u0 - self.u0_est)),
            tf.reduce_mean(tf.square(self.ph_v0 - self.v0_est)),
            tf.reduce_mean(tf.square(u_l - u_r)),
            tf.reduce_mean(tf.square(v_l - v_r)),
            tf.reduce_mean(tf.square(ux_l - ux_r)),
            tf.reduce_mean(tf.square(vx_l - vx_r)),
            tf.reduce_mean(tf.square(self.fu_est)),
            tf.reduce_mean(tf.square(self.fv_est)),
        ]
        self.objective = tf.add_n(loss_terms)

    def _configure_optimizers(self):
        lbfgs_opts = {
            'maxiter': 50000,
            'maxfun': 50000,
            'maxcor': 50,
            'maxls': 50,
            'ftol': 1.0 * np.finfo(float).eps,
        }
        self.lbfgs_opt = tf.contrib.opt.ScipyOptimizerInterface(
            self.objective, method='L-BFGS-B', options=lbfgs_opts
        )
        self.adam_step = tf.train.AdamOptimizer().minimize(self.objective)

    @staticmethod
    def _log_lbfgs(loss_val):
        print(f'Loss: {loss_val}')

    def fit(self, adam_steps):
        feed_map = {
            self.ph_x0: self.x0,
            self.ph_t0: self.t0,
            self.ph_u0: self.u0,
            self.ph_v0: self.v0,
            self.ph_x_left: self.x_left,
            self.ph_t_left: self.t_left,
            self.ph_x_right: self.x_right,
            self.ph_t_right: self.t_right,
            self.ph_x_col: self.x_col,
            self.ph_t_col: self.t_col,
        }

        t_mark = time.time()
        for step in range(adam_steps):
            self.session.run(self.adam_step, feed_map)
            if step % 10 == 0:
                elapsed = time.time() - t_mark
                cur_loss = self.session.run(self.objective, feed_map)
                print(f'It: {step}, Loss: {cur_loss:.3e}, Time: {elapsed:.2f}')
                t_mark = time.time()

        self.lbfgs_opt.minimize(
            self.session,
            feed_dict=feed_map,
            fetches=[self.objective],
            loss_callback=self._log_lbfgs,
        )

    def infer(self, query_pts):
        x_q = query_pts[:, 0:1]
        t_q = query_pts[:, 1:2]
        u_out, v_out = self.session.run([self.u0_est, self.v0_est], {self.ph_x0: x_q, self.ph_t0: t_q})
        fu_out, fv_out = self.session.run([self.fu_est, self.fv_est], {self.ph_x_col: x_q, self.ph_t_col: t_q})
        return u_out, v_out, fu_out, fv_out


def load_nls_dataset(mat_path, n_init, n_bnd, n_col, lb, ub):
    raw = scipy.io.loadmat(mat_path)
    t_grid = raw['tt'].flatten()[:, None]
    x_grid = raw['x'].flatten()[:, None]
    h_complex = raw['uu']

    u_exact = np.real(h_complex)
    v_exact = np.imag(h_complex)
    h_mag = np.sqrt(u_exact ** 2 + v_exact ** 2)

    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)
    xt_eval = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))

    idx_x = np.random.choice(x_grid.shape[0], n_init, replace=False)
    x_init = x_grid[idx_x, :]
    u_init = u_exact[idx_x, 0:1]
    v_init = v_exact[idx_x, 0:1]

    idx_t = np.random.choice(t_grid.shape[0], n_bnd, replace=False)
    t_bnd = t_grid[idx_t, :]

    col_pts = lb + (ub - lb) * lhs(2, n_col)
    return (
        t_grid, x_grid, u_exact, v_exact, h_mag,
        x_mesh, t_mesh, xt_eval, x_init, u_init, v_init, t_bnd, col_pts
    )


def plot_nls_results(t_grid, x_grid, h_exact, h_pred_grid, x_init, t_bnd, lb, ub):
    init_pts = np.concatenate((x_init, np.zeros_like(x_init)), axis=1)
    left_pts = np.concatenate((np.full_like(t_bnd, lb[0]), t_bnd), axis=1)
    right_pts = np.concatenate((np.full_like(t_bnd, ub[0]), t_bnd), axis=1)
    all_train_pts = np.vstack([init_pts, left_pts, right_pts])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')

    gs_top = gridspec.GridSpec(1, 2)
    gs_top.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax_map = plt.subplot(gs_top[:, :])

    img = ax_map.imshow(
        h_pred_grid.T,
        interpolation='nearest',
        cmap='YlGnBu',
        extent=[lb[1], ub[1], lb[0], ub[0]],
        origin='lower',
        aspect='auto',
    )
    cbar_ax = make_axes_locatable(ax_map).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cbar_ax)

    ax_map.plot(
        all_train_pts[:, 1],
        all_train_pts[:, 0],
        'kx',
        label=f'Data ({all_train_pts.shape[0]} points)',
        markersize=4,
        clip_on=False,
    )

    span = np.linspace(x_grid.min(), x_grid.max(), 2)[:, None]
    snap_indices = (75, 100, 125)
    for s_idx in snap_indices:
        ax_map.plot(t_grid[s_idx] * np.ones((2, 1)), span, 'k--', linewidth=1)

    ax_map.set_xlabel('$t$')
    ax_map.set_ylabel('$x$')
    ax_map.legend(frameon=False, loc='best')
    ax_map.set_title('$|h(t,x)|$', fontsize=10)

    gs_bot = gridspec.GridSpec(1, 3)
    gs_bot.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    for col, s_idx in enumerate(snap_indices):
        ax_s = plt.subplot(gs_bot[0, col])
        ax_s.plot(x_grid, h_exact[:, s_idx], 'b-', linewidth=2, label='Exact')
        ax_s.plot(x_grid, h_pred_grid[s_idx, :], 'r--', linewidth=2, label='Prediction')
        ax_s.set_xlabel('$x$')
        ax_s.set_ylabel('$|h(t,x)|$')
        ax_s.set_title(f'$t = {t_grid[s_idx].item():.2f}$', fontsize=10)
        ax_s.axis('square')
        ax_s.set_xlim([-5.1, 5.1])
        ax_s.set_ylim([-0.1, 5.1])
        if col == 1:
            ax_s.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    # savefig('./figures/NLS')


def main():
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2.0])
    n_init, n_bnd, n_col = 50, 50, 20000
    arch = [2, 100, 100, 100, 100, 2]

    (
        t_grid, x_grid, u_exact, v_exact, h_mag,
        x_mesh, t_mesh, xt_eval, x_init, u_init, v_init, t_bnd, col_pts
    ) = load_nls_dataset('../Data/NLS.mat', n_init, n_bnd, n_col, lb, ub)

    solver = NLSSolver(x_init, u_init, v_init, t_bnd, col_pts, arch, lb, ub)

    t_start = time.time()
    solver.fit(50000)
    print(f'Training time: {time.time() - t_start:.4f}')

    u_pred, v_pred, fu_pred, fv_pred = solver.infer(xt_eval)
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    u_true_flat = u_exact.T.flatten()[:, None]
    v_true_flat = v_exact.T.flatten()[:, None]
    h_true_flat = h_mag.T.flatten()[:, None]

    print(f'Error u: {np.linalg.norm(u_true_flat - u_pred, 2) / np.linalg.norm(u_true_flat, 2):e}')
    print(f'Error v: {np.linalg.norm(v_true_flat - v_pred, 2) / np.linalg.norm(v_true_flat, 2):e}')
    print(f'Error h: {np.linalg.norm(h_true_flat - h_pred, 2) / np.linalg.norm(h_true_flat, 2):e}')

    h_pred_grid = griddata(xt_eval, h_pred.flatten(), (x_mesh, t_mesh), method='cubic')
    plot_nls_results(t_grid, x_grid, h_mag, h_pred_grid, x_init, t_bnd, lb, ub)


if __name__ == '__main__':
    main()
