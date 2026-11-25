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


class AllenCahnSolver:
    def __init__(self, x_init, u_init, x_bnd, arch, delta_t, domain_min, domain_max, stages):
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.x_init = x_init
        self.u_init = u_init
        self.x_bnd = x_bnd
        self.arch = arch
        self.delta_t = delta_t
        self.stages = max(stages, 1)

        self.W_list, self.b_list = self._init_network_params(arch)
        self.irk_weights, self.irk_nodes = self._load_butcher_tableau(stages)

        session_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.session = tf.Session(config=session_cfg)

        self._build_graph()
        self._configure_optimizers()
        self.session.run(tf.global_variables_initializer())

    @staticmethod
    def _load_butcher_tableau(stages):
        path = f'../../Utilities/IRK_weights/Butcher_IRK{stages}.txt'
        raw_vals = np.float32(np.loadtxt(path, ndmin=2))
        split_idx = stages ** 2 + stages
        weight_mat = np.reshape(raw_vals[:split_idx], (stages + 1, stages))
        time_nodes = raw_vals[split_idx:]
        return weight_mat, time_nodes

    @staticmethod
    def _glorot_variable(fan_in, fan_out):
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        init_tensor = tf.truncated_normal([fan_in, fan_out], stddev=scale)
        return tf.Variable(init_tensor, dtype=tf.float32)

    def _init_network_params(self, arch):
        weights, biases = [], []
        for idx in range(len(arch) - 1):
            w_var = self._glorot_variable(arch[idx], arch[idx + 1])
            b_var = tf.Variable(tf.zeros([1, arch[idx + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(w_var)
            biases.append(b_var)
        return weights, biases

    def _forward_pass(self, coords):
        scaled = 2.0 * (coords - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        hidden = scaled
        for w_mat, b_vec in zip(self.W_list[:-1], self.b_list[:-1]):
            hidden = tf.tanh(tf.matmul(hidden, w_mat) + b_vec)
        return tf.matmul(hidden, self.W_list[-1]) + self.b_list[-1]

    @staticmethod
    def _forward_mode_grad(outputs, wrt_input, dummy_vec):
        first_pass = tf.gradients(outputs, wrt_input, grad_ys=dummy_vec)[0]
        return tf.gradients(first_pass, dummy_vec)[0]

    def _evaluate_initial_stages(self, x_tensor):
        stage_and_final = self._forward_pass(x_tensor)
        stage_vals = stage_and_final[:, :-1]
        du_dx = self._forward_mode_grad(stage_vals, x_tensor, self.ph_dummy_init)
        d2u_dx2 = self._forward_mode_grad(du_dx, x_tensor, self.ph_dummy_init)
        rhs_op = 5.0 * stage_vals - 5.0 * (stage_vals ** 3) + 0.0001 * d2u_dx2
        return stage_and_final - self.delta_t * tf.matmul(rhs_op, self.irk_weights.T)

    def _evaluate_boundary_state(self, x_tensor):
        state_vals = self._forward_pass(x_tensor)
        state_grad = self._forward_mode_grad(state_vals, x_tensor, self.ph_dummy_bnd)
        return state_vals, state_grad

    def _build_graph(self):
        self.ph_x_init = tf.placeholder(tf.float32, shape=(None, self.x_init.shape[1]))
        self.ph_x_bnd = tf.placeholder(tf.float32, shape=(None, self.x_bnd.shape[1]))
        self.ph_u_init = tf.placeholder(tf.float32, shape=(None, self.u_init.shape[1]))
        self.ph_dummy_init = tf.placeholder(tf.float32, shape=(None, self.stages))
        self.ph_dummy_bnd = tf.placeholder(tf.float32, shape=(None, self.stages + 1))

        self.u0_estimated = self._evaluate_initial_stages(self.ph_x_init)
        self.u1_estimated, self.u1_x_estimated = self._evaluate_boundary_state(self.ph_x_bnd)

        init_sse = tf.reduce_sum(tf.square(self.ph_u_init - self.u0_estimated))
        periodic_val_sse = tf.reduce_sum(tf.square(self.u1_estimated[0, :] - self.u1_estimated[1, :]))
        periodic_grad_sse = tf.reduce_sum(tf.square(self.u1_x_estimated[0, :] - self.u1_x_estimated[1, :]))
        self.objective = init_sse + periodic_val_sse + periodic_grad_sse

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
            self.ph_x_init: self.x_init,
            self.ph_u_init: self.u_init,
            self.ph_x_bnd: self.x_bnd,
            self.ph_dummy_init: np.ones((self.x_init.shape[0], self.stages)),
            self.ph_dummy_bnd: np.ones((self.x_bnd.shape[0], self.stages + 1)),
        }

        t_mark = time.time()
        for step in range(adam_steps):
            self.session.run(self.adam_step, feed_map)
            if step % 10 == 0:
                dt_elapsed = time.time() - t_mark
                cur_loss = self.session.run(self.objective, feed_map)
                print(f'It: {step}, Loss: {cur_loss:.3e}, Time: {dt_elapsed:.2f}')
                t_mark = time.time()

        self.lbfgs_opt.minimize(
            self.session,
            feed_dict=feed_map,
            fetches=[self.objective],
            loss_callback=self._log_lbfgs,
        )

    def infer(self, x_query):
        return self.session.run(self.u1_estimated, {self.ph_x_bnd: x_query})


def load_ac_dataset(mat_path, n_samples, t0_idx, t1_idx, noise_level=0.0):
    raw = scipy.io.loadmat(mat_path)
    t_grid = raw['tt'].flatten()[:, None]
    x_grid = raw['x'].flatten()[:, None]
    u_exact = np.real(raw['uu']).T

    delta_t = t_grid[t1_idx] - t_grid[t0_idx]
    sample_indices = np.random.choice(u_exact.shape[1], n_samples, replace=False)
    x_obs = x_grid[sample_indices, :]
    u_obs = u_exact[t0_idx:t0_idx + 1, sample_indices].T
    if noise_level > 0.0:
        u_obs = u_obs + noise_level * np.std(u_obs) * np.random.randn(*u_obs.shape)

    return t_grid, x_grid, u_exact, x_obs, u_obs, delta_t


def plot_ac_results(t_grid, x_grid, u_exact, x_obs, u_obs, u1_pred, t0_idx, t1_idx, lb, ub):
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')

    top_grid = gridspec.GridSpec(1, 2)
    top_grid.update(top=1 - 0.06, bottom=1 - 0.5 + 0.1, left=0.15, right=0.85, wspace=0)
    ax_map = plt.subplot(top_grid[:, :])

    img = ax_map.imshow(
        u_exact.T,
        interpolation='nearest',
        cmap='seismic',
        extent=[t_grid.min(), t_grid.max(), x_grid.min(), x_grid.max()],
        origin='lower',
        aspect='auto',
    )
    cbar_ax = make_axes_locatable(ax_map).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cbar_ax)

    vert_span = np.linspace(x_grid.min(), x_grid.max(), 2)[:, None]
    for tidx in (t0_idx, t1_idx):
        ax_map.plot(t_grid[tidx] * np.ones((2, 1)), vert_span, 'w-', linewidth=1)

    ax_map.set_xlabel('$t$')
    ax_map.set_ylabel('$x$')
    ax_map.set_title('$u(t,x)$', fontsize=10)

    bot_grid = gridspec.GridSpec(1, 2)
    bot_grid.update(top=0.5 - 0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    ax_t0 = plt.subplot(bot_grid[0, 0])
    ax_t0.plot(x_grid, u_exact[t0_idx, :], 'b-', linewidth=2)
    ax_t0.plot(x_obs, u_obs, 'rx', linewidth=2, label='Data')
    ax_t0.set_xlabel('$x$')
    ax_t0.set_ylabel('$u(t,x)$')
    ax_t0.set_title(f'$t = {t_grid[t0_idx].item():.2f}$', fontsize=10)
    ax_t0.set_xlim([lb - 0.1, ub + 0.1])
    ax_t0.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    ax_t1 = plt.subplot(bot_grid[0, 1])
    ax_t1.plot(x_grid, u_exact[t1_idx, :], 'b-', linewidth=2, label='Exact')
    ax_t1.plot(x_grid, u1_pred[:, -1], 'r--', linewidth=2, label='Prediction')
    ax_t1.set_xlabel('$x$')
    ax_t1.set_ylabel('$u(t,x)$')
    ax_t1.set_title(f'$t = {t_grid[t1_idx].item():.2f}$', fontsize=10)
    ax_t1.set_xlim([lb - 0.1, ub + 0.1])
    ax_t1.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)

    # savefig('./figures/AC')


def main():
    stages = 100
    arch = [1, 200, 200, 200, 200, stages + 1]
    lb = np.array([-1.0])
    ub = np.array([1.0])
    n_samples = 200
    t0_idx, t1_idx = 20, 180

    t_grid, x_grid, u_exact, x_obs, u_obs, delta_t = load_ac_dataset(
        '../Data/AC.mat', n_samples, t0_idx, t1_idx, noise_level=0.0
    )
    x_bnd = np.vstack((lb, ub))

    solver = AllenCahnSolver(x_obs, u_obs, x_bnd, arch, delta_t, lb, ub, stages)
    solver.fit(10000)

    u1_pred = solver.infer(x_grid)
    rel_l2 = np.linalg.norm(u1_pred[:, -1] - u_exact[t1_idx, :], 2) / np.linalg.norm(u_exact[t1_idx, :], 2)
    print(f'Error: {rel_l2:e}')

    plot_ac_results(t_grid, x_grid, u_exact, x_obs, u_obs, u1_pred, t0_idx, t1_idx, lb, ub)


if __name__ == '__main__':
    main()
