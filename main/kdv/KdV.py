import sys
import time
import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, '../../Utilities/')
from plotting import newfig, savefig

np.random.seed(1234)
tf.set_random_seed(1234)


class KdVInverseSolver:
    def __init__(self, x_t0, u_t0, x_t1, u_t1, arch, delta_t, domain_min, domain_max, stages):
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.x_t0 = x_t0
        self.u_t0 = u_t0
        self.x_t1 = x_t1
        self.u_t1 = u_t1
        self.arch = arch
        self.delta_t = delta_t
        self.stages = max(stages, 1)

        self.W_list, self.b_list = self._init_network_params(arch)
        self.coef_adv = tf.Variable([0.0], dtype=tf.float32)
        self.log_coef_disp = tf.Variable([-6.0], dtype=tf.float32)

        self.rk_alpha, self.rk_beta, self.rk_nodes = self._parse_irk_file(stages)

        session_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.session = tf.Session(config=session_cfg)

        self._build_graph()
        self._configure_optimizers()
        self.session.run(tf.global_variables_initializer())

    @staticmethod
    def _parse_irk_file(stages):
        path = f'../../Utilities/IRK_weights/Butcher_IRK{stages}.txt'
        raw_vals = np.float32(np.loadtxt(path, ndmin=2))
        split_idx = stages ** 2 + stages
        tableau = np.reshape(raw_vals[:split_idx], (stages + 1, stages))
        return tableau[:-1, :], tableau[-1:, :], raw_vals[split_idx:]

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

    def _forward_pass(self, coords):
        hidden = 2.0 * (coords - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        for w_mat, b_vec in zip(self.W_list[:-1], self.b_list[:-1]):
            hidden = tf.tanh(tf.matmul(hidden, w_mat) + b_vec)
        return tf.matmul(hidden, self.W_list[-1]) + self.b_list[-1]

    @staticmethod
    def _fwd_diff(outputs, wrt_input, dummy_vec):
        grad_step = tf.gradients(outputs, wrt_input, grad_ys=dummy_vec)[0]
        return tf.gradients(grad_step, dummy_vec)[0]

    def _kdv_operator(self, x_tensor, dummy_vec):
        l1 = self.coef_adv
        l2 = tf.exp(self.log_coef_disp)
        u_stages = self._forward_pass(x_tensor)
        u_x = self._fwd_diff(u_stages, x_tensor, dummy_vec)
        u_xx = self._fwd_diff(u_x, x_tensor, dummy_vec)
        u_xxx = self._fwd_diff(u_xx, x_tensor, dummy_vec)
        rhs = -l1 * u_stages * u_x - l2 * u_xxx
        return u_stages, rhs

    def _reconstruct_t0(self, x_tensor):
        u_stages, rhs = self._kdv_operator(x_tensor, self.ph_dummy0)
        return u_stages - self.delta_t * tf.matmul(rhs, self.rk_alpha.T)

    def _reconstruct_t1(self, x_tensor):
        u_stages, rhs = self._kdv_operator(x_tensor, self.ph_dummy1)
        return u_stages + self.delta_t * tf.matmul(rhs, (self.rk_beta - self.rk_alpha).T)

    def _build_graph(self):
        self.ph_x0 = tf.placeholder(tf.float32, shape=(None, self.x_t0.shape[1]))
        self.ph_x1 = tf.placeholder(tf.float32, shape=(None, self.x_t1.shape[1]))
        self.ph_u0 = tf.placeholder(tf.float32, shape=(None, self.u_t0.shape[1]))
        self.ph_u1 = tf.placeholder(tf.float32, shape=(None, self.u_t1.shape[1]))
        self.ph_dummy0 = tf.placeholder(tf.float32, shape=(None, self.stages))
        self.ph_dummy1 = tf.placeholder(tf.float32, shape=(None, self.stages))

        self.u0_estimated = self._reconstruct_t0(self.ph_x0)
        self.u1_estimated = self._reconstruct_t1(self.ph_x1)

        self.objective = (
            tf.reduce_sum(tf.square(self.ph_u0 - self.u0_estimated))
            + tf.reduce_sum(tf.square(self.ph_u1 - self.u1_estimated))
        )

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
            self.ph_x0: self.x_t0,
            self.ph_u0: self.u_t0,
            self.ph_x1: self.x_t1,
            self.ph_u1: self.u_t1,
            self.ph_dummy0: np.ones((self.x_t0.shape[0], self.stages)),
            self.ph_dummy1: np.ones((self.x_t1.shape[0], self.stages)),
        }

        t_mark = time.time()
        for step in range(adam_steps):
            self.session.run(self.adam_step, feed_map)
            if step % 10 == 0:
                elapsed = time.time() - t_mark
                cur_loss = self.session.run(self.objective, feed_map)
                l1_val, l2_val = self.get_discovered_params()
                print(
                    f'It: {step}, Loss: {cur_loss:.3e}, '
                    f'l1: {l1_val:.3f}, l2: {l2_val:.5f}, Time: {elapsed:.2f}'
                )
                t_mark = time.time()

        self.lbfgs_opt.minimize(
            self.session,
            feed_dict=feed_map,
            fetches=[self.objective],
            loss_callback=self._log_lbfgs,
        )

    def get_discovered_params(self):
        l1_val = self.session.run(self.coef_adv).item()
        l2_val = float(np.exp(self.session.run(self.log_coef_disp).item()))
        return l1_val, l2_val

    def infer(self, x_query):
        ones_mat = np.ones((x_query.shape[0], self.stages))
        u0_out = self.session.run(self.u0_estimated, {self.ph_x0: x_query, self.ph_dummy0: ones_mat})
        u1_out = self.session.run(self.u1_estimated, {self.ph_x1: x_query, self.ph_dummy1: ones_mat})
        return u0_out, u1_out


def load_kdv_dataset(mat_path, n0, n1, t_idx, step_skip):
    raw = scipy.io.loadmat(mat_path)
    t_grid = raw['tt'].flatten()[:, None]
    x_grid = raw['x'].flatten()[:, None]
    u_exact = np.real(raw['uu'])

    idx0 = np.random.choice(u_exact.shape[0], n0, replace=False)
    x0 = x_grid[idx0, :]
    u0 = u_exact[idx0, t_idx][:, None]

    idx1 = np.random.choice(u_exact.shape[0], n1, replace=False)
    x1 = x_grid[idx1, :]
    u1 = u_exact[idx1, t_idx + step_skip][:, None]

    delta_t = (t_grid[t_idx + step_skip] - t_grid[t_idx]).item()
    return t_grid, x_grid, u_exact, x0, u0, x1, u1, delta_t


def render_kdv_figure(
    t_grid, x_grid, u_exact, x0, u0, x1, u1, t_idx, step_skip, lb, ub, clean_params, noisy_params
):
    fig, ax = newfig(1.0, 1.5)
    ax.axis('off')

    gs_top = gridspec.GridSpec(1, 2)
    gs_top.update(top=1 - 0.06, bottom=1 - 1 / 3 + 0.05, left=0.15, right=0.85, wspace=0)
    ax_map = plt.subplot(gs_top[:, :])

    img = ax_map.imshow(
        u_exact,
        interpolation='nearest',
        cmap='rainbow',
        extent=[t_grid.min(), t_grid.max(), lb[0], ub[0]],
        origin='lower',
        aspect='auto',
    )
    cbar_ax = make_axes_locatable(ax_map).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cbar_ax)

    span = np.linspace(x_grid.min(), x_grid.max(), 2)[:, None]
    ax_map.plot(t_grid[t_idx] * np.ones((2, 1)), span, 'w-', linewidth=1.0)
    ax_map.plot(t_grid[t_idx + step_skip] * np.ones((2, 1)), span, 'w-', linewidth=1.0)
    ax_map.set_xlabel('$t$')
    ax_map.set_ylabel('$x$')
    ax_map.set_title('$u(t,x)$', fontsize=10)

    gs_mid = gridspec.GridSpec(1, 2)
    gs_mid.update(top=1 - 1 / 3 - 0.1, bottom=1 - 2 / 3, left=0.15, right=0.85, wspace=0.5)

    ax_s0 = plt.subplot(gs_mid[0, 0])
    ax_s0.plot(x_grid, u_exact[:, t_idx][:, None], 'b', linewidth=2, label='Exact')
    ax_s0.plot(x0, u0, 'rx', linewidth=2, label='Data')
    ax_s0.set_xlabel('$x$')
    ax_s0.set_ylabel('$u(t,x)$')
    ax_s0.set_title(f'$t = {t_grid[t_idx].item():.2f}$\n{u0.shape[0]} training data', fontsize=10)

    ax_s1 = plt.subplot(gs_mid[0, 1])
    ax_s1.plot(x_grid, u_exact[:, t_idx + step_skip][:, None], 'b', linewidth=2, label='Exact')
    ax_s1.plot(x1, u1, 'rx', linewidth=2, label='Data')
    ax_s1.set_xlabel('$x$')
    ax_s1.set_ylabel('$u(t,x)$')
    ax_s1.set_title(f'$t = {t_grid[t_idx + step_skip].item():.2f}$\n{u1.shape[0]} training data', fontsize=10)
    ax_s1.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

    gs_bot = gridspec.GridSpec(1, 2)
    gs_bot.update(top=1 - 2 / 3 - 0.05, bottom=0, left=0.15, right=0.85, wspace=0.0)
    ax_tbl = plt.subplot(gs_bot[0, 0])
    ax_tbl.axis('off')

    l1_c, l2_c = clean_params
    l1_n, l2_n = noisy_params
    latex_table = (
        r'$\begin{tabular}{ |c|c| } \hline '
        r'Correct PDE & $u_t + u u_x + 0.0025 u_{xxx} = 0$ \\ \hline '
        f'Identified PDE (clean data) & $u_t + {l1_c:.3f} u u_x + {l2_c:.7f} u_{{xxx}} = 0$ \\\\ \\hline '
        f'Identified PDE (1\\% noise) & $u_t + {l1_n:.3f} u u_x + {l2_n:.7f} u_{{xxx}} = 0$ \\\\ \\hline '
        r'\end{tabular}$'
    )
    ax_tbl.text(-0.1, 0.2, latex_table)

    # savefig('./figures/KdV')


def main():
    stages = 50
    step_skip = 120
    n0, n1 = 199, 201
    t_idx = 40
    arch = [1, 50, 50, 50, 50, stages]

    t_grid, x_grid, u_exact, x0, u0, x1, u1, delta_t = load_kdv_dataset(
        '../Data/KdV.mat', n0, n1, t_idx, step_skip
    )
    lb = x_grid.min(0)
    ub = x_grid.max(0)

    clean_solver = KdVInverseSolver(x0, u0, x1, u1, arch, delta_t, lb, ub, stages)
    clean_solver.fit(adam_steps=50000)
    clean_solver.infer(x_grid)

    l1_clean, l2_clean = clean_solver.get_discovered_params()
    err_l1 = abs(l1_clean - 1.0) / 1.0 * 100.0
    err_l2 = abs(l2_clean - 0.0025) / 0.0025 * 100.0
    print(f'Error lambda_1: {err_l1:f}%')
    print(f'Error lambda_2: {err_l2:f}%')

    noise_ratio = 0.01
    u0_noisy = u0 + noise_ratio * np.std(u0) * np.random.randn(*u0.shape)
    u1_noisy = u1 + noise_ratio * np.std(u1) * np.random.randn(*u1.shape)

    noisy_solver = KdVInverseSolver(x0, u0_noisy, x1, u1_noisy, arch, delta_t, lb, ub, stages)
    noisy_solver.fit(adam_steps=50000)
    noisy_solver.infer(x_grid)

    l1_noisy, l2_noisy = noisy_solver.get_discovered_params()
    err_l1_n = abs(l1_noisy - 1.0) / 1.0 * 100.0
    err_l2_n = abs(l2_noisy - 0.0025) / 0.0025 * 100.0
    print(f'Error lambda_1: {err_l1_n:f}%')
    print(f'Error lambda_2: {err_l2_n:f}%')

    render_kdv_figure(
        t_grid, x_grid, u_exact, x0, u0_noisy, x1, u1_noisy,
        t_idx, step_skip, lb, ub, (l1_clean, l2_clean), (l1_noisy, l2_noisy)
    )


if __name__ == '__main__':
    main()
