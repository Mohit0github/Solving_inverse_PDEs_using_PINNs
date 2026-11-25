import sys
import time
from itertools import product, combinations
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, '../../Utilities/')
from plotting import newfig, savefig

np.random.seed(1234)
tf.set_random_seed(1234)


class NavierStokesInverseSolver:
    def __init__(self, x_obs, y_obs, t_obs, u_obs, v_obs, arch):
        coords = np.concatenate([x_obs, y_obs, t_obs], axis=1)
        self.domain_min = coords.min(0)
        self.domain_max = coords.max(0)

        self.x_obs = coords[:, 0:1]
        self.y_obs = coords[:, 1:2]
        self.t_obs = coords[:, 2:3]
        self.u_obs = u_obs
        self.v_obs = v_obs
        self.arch = arch

        self.W_list, self.b_list = self._init_network_params(arch)
        self.coef_conv = tf.Variable([0.0], dtype=tf.float32)
        self.coef_visc = tf.Variable([0.0], dtype=tf.float32)

        session_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.session = tf.Session(config=session_cfg)

        self._build_graph()
        self._configure_optimizers()
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

    def _compute_flow_residuals(self, x_tf, y_tf, t_tf):
        l1 = self.coef_conv
        l2 = self.coef_visc

        latent = self._forward_pass(tf.concat([x_tf, y_tf, t_tf], axis=1))
        stream_fn = latent[:, 0:1]
        pressure = latent[:, 1:2]

        vel_u = tf.gradients(stream_fn, y_tf)[0]
        vel_v = -tf.gradients(stream_fn, x_tf)[0]

        u_t = tf.gradients(vel_u, t_tf)[0]
        u_x = tf.gradients(vel_u, x_tf)[0]
        u_y = tf.gradients(vel_u, y_tf)[0]
        u_xx = tf.gradients(u_x, x_tf)[0]
        u_yy = tf.gradients(u_y, y_tf)[0]

        v_t = tf.gradients(vel_v, t_tf)[0]
        v_x = tf.gradients(vel_v, x_tf)[0]
        v_y = tf.gradients(vel_v, y_tf)[0]
        v_xx = tf.gradients(v_x, x_tf)[0]
        v_yy = tf.gradients(v_y, y_tf)[0]

        p_x = tf.gradients(pressure, x_tf)[0]
        p_y = tf.gradients(pressure, y_tf)[0]

        res_u = u_t + l1 * (vel_u * u_x + vel_v * u_y) + p_x - l2 * (u_xx + u_yy)
        res_v = v_t + l1 * (vel_u * v_x + vel_v * v_y) + p_y - l2 * (v_xx + v_yy)
        return vel_u, vel_v, pressure, res_u, res_v

    def _build_graph(self):
        self.ph_x = tf.placeholder(tf.float32, shape=[None, self.x_obs.shape[1]])
        self.ph_y = tf.placeholder(tf.float32, shape=[None, self.y_obs.shape[1]])
        self.ph_t = tf.placeholder(tf.float32, shape=[None, self.t_obs.shape[1]])
        self.ph_u = tf.placeholder(tf.float32, shape=[None, self.u_obs.shape[1]])
        self.ph_v = tf.placeholder(tf.float32, shape=[None, self.v_obs.shape[1]])

        self.u_est, self.v_est, self.p_est, self.fu_est, self.fv_est = self._compute_flow_residuals(
            self.ph_x, self.ph_y, self.ph_t
        )

        self.objective = (
            tf.reduce_sum(tf.square(self.ph_u - self.u_est))
            + tf.reduce_sum(tf.square(self.ph_v - self.v_est))
            + tf.reduce_sum(tf.square(self.fu_est))
            + tf.reduce_sum(tf.square(self.fv_est))
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
    def _log_lbfgs(loss_val, l1_val, l2_val):
        print(f'Loss: {loss_val:.3e}, l1: {float(l1_val):.3f}, l2: {float(l2_val):.5f}')

    def fit(self, adam_steps):
        feed_map = {
            self.ph_x: self.x_obs,
            self.ph_y: self.y_obs,
            self.ph_t: self.t_obs,
            self.ph_u: self.u_obs,
            self.ph_v: self.v_obs,
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
            fetches=[self.objective, self.coef_conv, self.coef_visc],
            loss_callback=self._log_lbfgs,
        )

    def get_discovered_params(self):
        l1_val = self.session.run(self.coef_conv).item()
        l2_val = self.session.run(self.coef_visc).item()
        return l1_val, l2_val

    def infer(self, x_q, y_q, t_q):
        feed_map = {self.ph_x: x_q, self.ph_y: y_q, self.ph_t: t_q}
        return self.session.run([self.u_est, self.v_est, self.p_est], feed_map)


def set_equal_3d_axes(ax):
    bounds = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    spans = bounds[:, 1] - bounds[:, 0]
    mids = np.mean(bounds, axis=1)
    half_range = max(abs(spans)) / 4.0
    for mid, axis in zip(mids, 'xyz'):
        getattr(ax, f'set_{axis}lim')(mid - half_range, mid + half_range)


def draw_3d_box_scatter(ax, x_pts, t_pts, y_pts, x_mesh, y_mesh, field_grid, x_bounds, t_bounds, y_bounds, slice_t, label_str):
    ax.axis('off')
    corners = np.array(list(product(x_bounds, t_bounds, y_bounds)))
    for p1, p2 in combinations(corners, 2):
        dist = np.sum(np.abs(p1 - p2))
        if (
            dist == x_bounds[1] - x_bounds[0]
            or dist == t_bounds[1] - t_bounds[0]
            or dist == y_bounds[1] - y_bounds[0]
        ):
            ax.plot3D(*zip(p1, p2), color='k', linewidth=0.5)

    ax.scatter(x_pts, t_pts, y_pts, s=0.1)
    ax.contourf(x_mesh, field_grid, y_mesh, zdir='y', offset=slice_t, cmap='rainbow', alpha=0.8)

    ax.text(np.mean(x_bounds), t_bounds[0] - 1, y_bounds[0] - 1, '$x$')
    ax.text(x_bounds[1] + 1, np.mean(t_bounds), y_bounds[0] - 1, '$t$')
    ax.text(x_bounds[0] - 1, t_bounds[0] - 0.5, np.mean(y_bounds), '$y$')
    ax.text(x_bounds[0] - 3, np.mean(t_bounds), y_bounds[1] + 1, label_str)
    ax.set_xlim3d(x_bounds)
    ax.set_ylim3d(t_bounds)
    ax.set_zlim3d(y_bounds)
    set_equal_3d_axes(ax)


def load_wake_dataset(mat_path, n_train):
    raw = scipy.io.loadmat(mat_path)
    vel_data = raw['U_star']
    pres_data = raw['p_star']
    t_vec = raw['t']
    xy_coords = raw['X_star']

    n_nodes = xy_coords.shape[0]
    n_times = t_vec.shape[0]

    xx_mat = np.tile(xy_coords[:, 0:1], (1, n_times))
    yy_mat = np.tile(xy_coords[:, 1:2], (1, n_times))
    tt_mat = np.tile(t_vec, (1, n_nodes)).T

    x_flat = xx_mat.flatten()[:, None]
    y_flat = yy_mat.flatten()[:, None]
    t_flat = tt_mat.flatten()[:, None]
    u_flat = vel_data[:, 0, :].flatten()[:, None]
    v_flat = vel_data[:, 1, :].flatten()[:, None]

    sample_idx = np.random.choice(n_nodes * n_times, n_train, replace=False)
    train_tuple = (
        x_flat[sample_idx, :],
        y_flat[sample_idx, :],
        t_flat[sample_idx, :],
        u_flat[sample_idx, :],
        v_flat[sample_idx, :],
    )
    return xy_coords, t_vec, vel_data, pres_data, tt_mat, train_tuple


def plot_ns_results(
    vort_path, xy_coords, t_vec, x_tr, t_tr, y_tr, t_snap,
    x_mesh, y_mesh, u_grid, v_grid, p_pred_grid, p_exact_grid,
    clean_params, noisy_params
):
    vort_raw = scipy.io.loadmat(vort_path)
    modes = vort_raw['modes'].item()
    n_elem = vort_raw['nel'].item()
    shape_v = (modes + 1, modes + 1, n_elem)

    xv = np.reshape(vort_raw['x'], shape_v, order='F')
    yv = np.reshape(vort_raw['y'], shape_v, order='F')
    wv = np.reshape(vort_raw['w'], shape_v, order='F')

    roi_min = np.array([1.0, -2.0])
    roi_max = np.array([8.0, 2.0])

    fig1, ax1 = newfig(1.0, 1.2)
    ax1.axis('off')

    gs_vort = gridspec.GridSpec(1, 2)
    gs_vort.update(top=1 - 0.06, bottom=0.5 + 0.12, left=0.0, right=1.0, wspace=0)
    ax_v = plt.subplot(gs_vort[:, :])
    for elem in range(n_elem):
        mesh_h = ax_v.pcolormesh(
            xv[:, :, elem], yv[:, :, elem], wv[:, :, elem],
            cmap='seismic', shading='gouraud', vmin=-3, vmax=3
        )
    cbar_ax = make_axes_locatable(ax_v).append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(mesh_h, cax=cbar_ax)

    ax_v.plot([roi_min[0], roi_min[0]], [roi_min[1], roi_max[1]], 'k', linewidth=1)
    ax_v.plot([roi_max[0], roi_max[0]], [roi_min[1], roi_max[1]], 'k', linewidth=1)
    ax_v.plot([roi_min[0], roi_max[0]], [roi_min[1], roi_min[1]], 'k', linewidth=1)
    ax_v.plot([roi_min[0], roi_max[0]], [roi_max[1], roi_max[1]], 'k', linewidth=1)
    ax_v.set_aspect('equal', 'box')
    ax_v.set_xlabel('$x$')
    ax_v.set_ylabel('$y$')
    ax_v.set_title('Vorticity', fontsize=10)

    gs_3d = gridspec.GridSpec(1, 2)
    gs_3d.update(top=0.5, bottom=0.0, left=0.01, right=0.99, wspace=0)
    x_rng = [xy_coords[:, 0].min(), xy_coords[:, 0].max()]
    t_rng = [t_vec.min(), t_vec.max()]
    y_rng = [xy_coords[:, 1].min(), xy_coords[:, 1].max()]

    ax_u3d = plt.subplot(gs_3d[:, 0], projection='3d')
    draw_3d_box_scatter(ax_u3d, x_tr, t_tr, y_tr, x_mesh, y_mesh, u_grid, x_rng, t_rng, y_rng, t_snap.mean(), '$u(t,x,y)$')

    ax_v3d = plt.subplot(gs_3d[:, 1], projection='3d')
    draw_3d_box_scatter(ax_v3d, x_tr, t_tr, y_tr, x_mesh, y_mesh, v_grid, x_rng, t_rng, y_rng, t_snap.mean(), '$v(t,x,y)$')
    # savefig('./figures/NavierStokes_data')

    fig2, ax2 = newfig(1.015, 0.8)
    ax2.axis('off')

    gs_pres = gridspec.GridSpec(1, 2)
    gs_pres.update(top=1, bottom=0.5, left=0.1, right=0.9, wspace=0.5)
    extent_xy = [x_rng[0], x_rng[1], y_rng[0], y_rng[1]]

    for col_idx, (grid_data, title_str) in enumerate([(p_pred_grid, 'Predicted pressure'), (p_exact_grid, 'Exact pressure')]):
        ax_p = plt.subplot(gs_pres[:, col_idx])
        im_p = ax_p.imshow(grid_data, interpolation='nearest', cmap='rainbow', extent=extent_xy, origin='lower', aspect='auto')
        cax_p = make_axes_locatable(ax_p).append_axes('right', size='5%', pad=0.05)
        fig2.colorbar(im_p, cax=cax_p)
        ax_p.set_xlabel('$x$')
        ax_p.set_ylabel('$y$')
        ax_p.set_aspect('equal', 'box')
        ax_p.set_title(title_str, fontsize=10)

    gs_tbl = gridspec.GridSpec(1, 2)
    gs_tbl.update(top=0.5, bottom=0.0, left=0.0, right=1.0, wspace=0)
    ax_t = plt.subplot(gs_tbl[:, :])
    ax_t.axis('off')

    l1_c, l2_c = clean_params
    l1_n, l2_n = noisy_params
    tbl_str = (
        r'$\begin{tabular}{|c|c|} \hline '
        r'Correct PDE & $\begin{array}{c} '
        r'u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy}) \\ '
        r'v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy}) '
        r'\end{array}$ \\ \hline '
        r'Identified PDE (clean data) & $\begin{array}{c} '
        f'u_t + {l1_c:.3f} (u u_x + v u_y) = -p_x + {l2_c:.5f} (u_{{xx}} + u_{{yy}}) \\\\ '
        f'v_t + {l1_c:.3f} (u v_x + v v_y) = -p_y + {l2_c:.5f} (v_{{xx}} + v_{{yy}}) '
        r'\end{array}$ \\ \hline '
        r'Identified PDE (1\% noise) & $\begin{array}{c} '
        f'u_t + {l1_n:.3f} (u u_x + v u_y) = -p_x + {l2_n:.5f} (u_{{xx}} + u_{{yy}}) \\\\ '
        f'v_t + {l1_n:.3f} (u v_x + v v_y) = -p_y + {l2_n:.5f} (v_{{xx}} + v_{{yy}}) '
        r'\end{array}$ \\ \hline \end{tabular}$'
    )
    ax_t.text(0.015, 0.0, tbl_str)
    # savefig('./figures/NavierStokes_prediction')


def main():
    n_train = 5000
    arch = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    xy_coords, t_vec, vel_data, pres_data, tt_mat, train_tuple = load_wake_dataset(
        '../Data/cylinder_nektar_wake.mat', n_train
    )
    x_tr, y_tr, t_tr, u_tr, v_tr = train_tuple

    clean_solver = NavierStokesInverseSolver(x_tr, y_tr, t_tr, u_tr, v_tr, arch)
    clean_solver.fit(200000)

    snap_idx = np.array([100])
    x_eval = xy_coords[:, 0:1]
    y_eval = xy_coords[:, 1:2]
    t_eval = tt_mat[:, snap_idx]
    u_true = vel_data[:, 0, snap_idx]
    v_true = vel_data[:, 1, snap_idx]
    p_true = pres_data[:, snap_idx]

    u_pred, v_pred, p_pred = clean_solver.infer(x_eval, y_eval, t_eval)
    l1_clean, l2_clean = clean_solver.get_discovered_params()

    err_u = np.linalg.norm(u_true - u_pred, 2) / np.linalg.norm(u_true, 2)
    err_v = np.linalg.norm(v_true - v_pred, 2) / np.linalg.norm(v_true, 2)
    err_p = np.linalg.norm(p_true - p_pred, 2) / np.linalg.norm(p_true, 2)
    print(f'Error u: {err_u:e}')
    print(f'Error v: {err_v:e}')
    print(f'Error p: {err_p:e}')
    print(f'Error l1: {abs(l1_clean - 1.0) * 100.0:.5f}%')
    print(f'Error l2: {abs(l2_clean - 0.01) / 0.01 * 100.0:.5f}%')

    lb = xy_coords.min(0)
    ub = xy_coords.max(0)
    grid_res = 200
    x_mesh, y_mesh = np.meshgrid(np.linspace(lb[0], ub[0], grid_res), np.linspace(lb[1], ub[1], grid_res))
    u_grid = griddata(xy_coords, u_pred.flatten(), (x_mesh, y_mesh), method='cubic')
    v_grid = griddata(xy_coords, v_pred.flatten(), (x_mesh, y_mesh), method='cubic')
    p_pred_grid = griddata(xy_coords, p_pred.flatten(), (x_mesh, y_mesh), method='cubic')
    p_exact_grid = griddata(xy_coords, p_true.flatten(), (x_mesh, y_mesh), method='cubic')

    noise_ratio = 0.01
    u_tr_noisy = u_tr + noise_ratio * np.std(u_tr) * np.random.randn(*u_tr.shape)
    v_tr_noisy = v_tr + noise_ratio * np.std(v_tr) * np.random.randn(*v_tr.shape)

    noisy_solver = NavierStokesInverseSolver(x_tr, y_tr, t_tr, u_tr_noisy, v_tr_noisy, arch)
    noisy_solver.fit(200000)
    l1_noisy, l2_noisy = noisy_solver.get_discovered_params()
    print(f'Error l1: {abs(l1_noisy - 1.0) * 100.0:.5f}%')
    print(f'Error l2: {abs(l2_noisy - 0.01) / 0.01 * 100.0:.5f}%')

    plot_ns_results(
        '../Data/cylinder_nektar_t0_vorticity.mat',
        xy_coords, t_vec, x_tr, t_tr, y_tr, t_eval,
        x_mesh, y_mesh, u_grid, v_grid, p_pred_grid, p_exact_grid,
        (l1_clean, l2_clean), (l1_noisy, l2_noisy),
    )


if __name__ == '__main__':
    main()
