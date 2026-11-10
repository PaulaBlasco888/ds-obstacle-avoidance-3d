"""
Builds the demonstration video (figures/demo_video.mp4, 1280x720, ~2 min)
with on-screen explanatory text. Every scene is simulated on the fly with the
same code used for the report figures.

    python make_video.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg

from ds_learning.lpvds import LPVDS
from main_3d import build_scenario, initial_points, integrate, set_equal_3d
from robot_sim import PandaDSController, QDOT_MAX

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
FPS = 25
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams['font.size'] = 12
OBS_COLOR = (0.69, 0.49, 0.49)
COLORS = plt.cm.tab10(np.arange(10))


def caption(fig, title, text):
    fig.text(0.5, 0.965, title, ha='center', va='top', fontsize=18, weight='bold')
    fig.text(0.5, 0.025, text, ha='center', va='bottom', fontsize=12.5,
             bbox=dict(boxstyle='round,pad=0.5', fc='#f3f3f7', ec='#bbbbcc'))


def hold(writer, fig, seconds):
    for _ in range(int(seconds * FPS)):
        writer.grab_frame()


# --------------------------------------------------------------------------- #
def scene_card(writer, fig, lines, seconds, big=None):
    fig.clf()
    y = 0.62 if big else 0.55
    if big:
        fig.text(0.5, 0.72, big, ha='center', fontsize=28, weight='bold')
    for i, (txt, size) in enumerate(lines):
        fig.text(0.5, y - i * 0.075, txt, ha='center', fontsize=size)
    hold(writer, fig, seconds)


def scene_image(writer, fig, path, title, text, seconds):
    fig.clf()
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.78])
    ax.imshow(mpimg.imread(path)); ax.axis('off')
    caption(fig, title, text)
    hold(writer, fig, seconds)


def scene_2d(writer, fig):
    """Learned 2D LPV-DS (S-shape) modulated around an obstacle (Lab Session 2 code)."""
    from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle
    from session2_lpvds_obstacle_2d import integrate as integrate2d, SCENARIOS, make_obstacles
    from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import obs_avoidance_interpolation_moving
    ds = LPVDS.load(os.path.join(HERE, 'models', 'sshape2d.json'))
    demo = np.load(os.path.join(HERE, 'models', 'sshape2d_data.npz'))
    sc = SCENARIOS['sshape2d']
    obs = make_obstacles(sc)
    for o in obs:
        o.draw_ellipsoid(numPoints=60)
    xs = np.linspace(*sc['xr'], 45); ys = np.linspace(*sc['yr'], 45)
    XX, YY = np.meshgrid(xs, ys); U = np.zeros_like(XX); V = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            x = np.array([XX[i, j], YY[i, j]])
            if np.sum(((x - obs[0].x0) / (np.array(obs[0].a))) ** 2) < 1:
                U[i, j] = V[i, j] = np.nan; continue
            v = obs_avoidance_interpolation_moving(x, ds(x), obs)
            U[i, j], V[i, j] = v
    starts = list(demo['x0_all'].T) + [x + d for x, d in zip(demo['x0_all'].T, [[0.1, -0.15], [-0.1, 0.1], [0.05, 0.12]])]
    trajs = [integrate2d(ds, x0, obs) for x0 in starts]

    fig.clf()
    ax = fig.add_axes([0.08, 0.13, 0.84, 0.75])
    ax.streamplot(XX, YY, U, V, color=[0.75, 0.75, 0.9], density=1.4, zorder=0)
    ax.add_patch(plt.Polygon(obs[0].x_obs_sf, color=[0.9, 0.9, 0.9], zorder=1))
    ax.add_patch(plt.Polygon(obs[0].x_obs, color=OBS_COLOR, zorder=2))
    ax.plot(demo['Data'][0], demo['Data'][1], 'r.', ms=1.5, alpha=0.5, label='demonstrations')
    ax.plot(*ds.attractor, 'k*', ms=18)
    ax.set_xlim(sc['xr']); ax.set_ylim(sc['yr']); ax.set_aspect('equal')
    caption(fig, '1 · Learned DS + obstacle avoidance in 2D (Lab Session 2)',
            'The linear attractor of the Lab 2 code is replaced by an LPV-DS learned from the S-shape demonstrations (red).\n'
            r'Trajectories keep the learned S-shape, slide around the obstacle ($\dot x = M(x)f(x)$) and converge to the target.')
    lines = [ax.plot([], [], lw=2.5, color=COLORS[k])[0] for k in range(len(trajs))]
    dots = [ax.plot([], [], 'o', color=COLORS[k], ms=7)[0] for k in range(len(trajs))]
    n = max(t.shape[1] for t in trajs)
    n_frames = 12 * FPS
    hold(writer, fig, 1.5)
    for f in range(n_frames):
        i = int(f / (n_frames - 1) * (n - 1))
        for ln, dt, tr in zip(lines, dots, trajs):
            j = min(i, tr.shape[1] - 1)
            ln.set_data(tr[0, :j + 1], tr[1, :j + 1]); dt.set_data([tr[0, j]], [tr[1, j]])
        writer.grab_frame()
    hold(writer, fig, 2)


def scene_3d(writer, fig, ds, obs, starts, Data):
    runs = {n: integrate(ds, x0, obs)[0] for n, x0 in starts.items()}
    frees = {n: integrate(ds, x0, None)[0] for n, x0 in starts.items()}
    fig.clf()
    ax = fig.add_axes([0.0, 0.1, 1.0, 0.8], projection='3d')
    X, Y, Z = obs.surface()
    ax.plot_surface(X, Y, Z, color=OBS_COLOR, alpha=0.85, linewidth=0)
    ax.scatter(*Data[:3, ::3], s=2, c='r', alpha=0.25)
    ax.scatter(*ds.attractor, marker='*', s=300, c='k')
    for k, n in enumerate(runs):
        ax.plot(*frees[n], '--', color=COLORS[k], lw=1, alpha=0.6)
    allp = np.hstack(list(runs.values()))
    set_equal_3d(ax, allp)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    caption(fig, '2 · 3D: learned LPV-DS modulated around an ellipsoid',
            r'$\Gamma(x)$, $n(x)=\nabla\Gamma$, orthonormal basis $[n,e_1,e_2]$, $D(x)$ and $M(x)=E D E^{-1}\in\mathbb{R}^{3\times3}$.  '
            'Dashed: learned DS without obstacle (goes through it).\n'
            'Solid: modulated DS from 5 starts (centre, above, below, left, right). All avoid the obstacle and reach the target.')
    lines = {n: ax.plot([], [], [], lw=2.5, color=COLORS[k], label=n)[0] for k, n in enumerate(runs)}
    dots = {n: ax.plot([], [], [], 'o', color=COLORS[k], ms=7)[0] for k, n in enumerate(runs)}
    ax.legend(loc='upper left')
    nmax = max(r.shape[1] for r in runs.values())
    n_frames = 22 * FPS
    for f in range(n_frames):
        i = int(min(1.0, f / (0.8 * n_frames)) * (nmax - 1))
        for n, tr in runs.items():
            j = min(i, tr.shape[1] - 1)
            lines[n].set_data(tr[0, :j + 1], tr[1, :j + 1]); lines[n].set_3d_properties(tr[2, :j + 1])
            dots[n].set_data([tr[0, j]], [tr[1, j]]); dots[n].set_3d_properties([tr[2, j]])
        ax.view_init(elev=22, azim=-70 + 70 * f / n_frames)
        writer.grab_frame()
    # zoom on the obstacle at the end
    for f in range(4 * FPS):
        s = f / (4 * FPS)
        box = np.column_stack([obs.c - (0.5 - 0.35 * s), obs.c + (0.5 - 0.35 * s)])
        set_equal_3d(ax, box)
        ax.view_init(elev=22 - 8 * s, azim=0 - 35 * s)
        writer.grab_frame()
    hold(writer, fig, 2)


def scene_robot(writer, fig, ds, obs, starts, Data):
    t_task = np.array([0.50, 0.0, 0.40]) - Data[:3].mean(axis=1)
    ctrl = PandaDSController(ds, obs, t_task)
    sel = ['centre', 'above', 'right']
    runs = {n: ctrl.run(ctrl.to_robot(starts[n])) for n in sel}
    fig.clf()
    ax = fig.add_axes([0.0, 0.1, 1.0, 0.8], projection='3d')
    X, Y, Z = obs.surface()
    ax.plot_surface(X + t_task[0], Y + t_task[1], Z + t_task[2], color=OBS_COLOR, alpha=0.85, linewidth=0)
    ax.scatter(*ctrl.to_robot(ds.attractor), marker='*', s=300, c='k')
    ax.plot([0], [0], [0], 's', color='k', ms=12)
    allp = np.vstack([np.vstack([r['p'] for r in runs.values()]), [[0, 0, 0]]]).T
    set_equal_3d(ax, allp)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    caption(fig, '3 · Franka Emika Panda driven by the modulated DS',
            r'Every 10 ms: $p=FK(q)$,  $\dot x_{mod}=M(x)f(x)$,  $\dot q = J^{+}(q)\,[\dot x_{mod};\ \omega]$ (damped least squares, gripper kept vertical,'
            '\nnull-space joint centring, real Panda velocity limits). Joints stay > 35° from their limits; |q̇| < 35 % of the limit.')
    title = fig.text(0.5, 0.9, '', ha='center', fontsize=13)
    arm = []
    trail_lines = {}
    for k, n in enumerate(sel):
        r = runs[n]
        col = COLORS[[0, 1, 4][k]]
        trail_lines[n] = ax.plot([], [], [], lw=2.5, color=col)[0]
        n_frames = int(len(r['q']) / 3)        # 3 control steps per frame -> 0.75x real time... ~1.2x speed
        for f in range(n_frames):
            i = min(f * 3, len(r['q']) - 1)
            for h in arm:
                h.remove()
            arm.clear()
            P = ctrl.joint_positions(r['q'][i])
            arm.append(ax.plot(P[:, 0], P[:, 1], P[:, 2], '-', color='0.3', lw=7, solid_capstyle='round')[0])
            arm.append(ax.plot(P[:, 0], P[:, 1], P[:, 2], 'o', color='k', ms=5)[0])
            trail_lines[n].set_data(r['p'][:i + 1, 0], r['p'][:i + 1, 1]); trail_lines[n].set_3d_properties(r['p'][:i + 1, 2])
            title.set_text(f'run "{n}"   t = {r["t"][i]:.2f} s    Γ = {r["gamma"][i]:.2f}')
            ax.view_init(elev=20, azim=-60 + 25 * np.sin(2 * np.pi * (f / max(n_frames, 1)) * 0.5 + k))
            writer.grab_frame()
        hold(writer, fig, 0.8)
    return runs, ctrl


def main():
    ds = LPVDS.load(os.path.join(HERE, 'models', 'viapoint3d.json'))
    demo = np.load(os.path.join(HERE, 'models', 'viapoint3d_data.npz'))
    Data, x0_all = demo['Data'], demo['x0_all']
    mean0 = x0_all.mean(axis=1)
    x_start = x0_all[:, np.argmin(np.linalg.norm(x0_all - mean0[:, None], axis=0))]
    obs, _ = build_scenario(ds, x_start)
    starts = initial_points(x_start, ds)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=FPS, bitrate=4000)
    out = os.path.join(FIG, 'demo_video.mp4')
    with writer.saving(fig, out, dpi=100):
        scene_card(writer, fig, [
            ('Robot\'s Control Architectures · UC3M', 16),
            ('Learned LPV-DS (Lab 1) + reactive modulation $\\dot x = M(x)\\,f(x)$ (Lab 2)', 15),
            ('2D integration  →  3D obstacle  →  Franka Emika Panda', 15)],
            6, big='3D Obstacle Avoidance with Dynamical Systems')
        scene_image(writer, fig, os.path.join(FIG, 'sshape2d_lpvds.png'),
                    '0 · Learning the nominal DS from demonstrations',
                    r'$f(x)=\sum_k \gamma_k(x)(A_k x+b_k)$: GMM for $\gamma_k$ + convex SDP for $A_k$ with $A_k^TP+PA_k\prec 0$, $b_k=-A_kx^*$'
                    '\n' r'$\Rightarrow V(x)=(x-x^*)^TP(x-x^*)$ decreases everywhere: globally asymptotically stable at the target $x^*$.', 8)
        scene_2d(writer, fig)
        scene_3d(writer, fig, ds, obs, starts, Data)
        scene_image(writer, fig, os.path.join(FIG, 'viapoint3d_lyapunov.png'),
                    '2b · Stability check of the 3D DS',
                    r'Left: $V(x(t))$ decreases monotonically along the nominal trajectories.  '
                    r'Right: $\dot V(x)<0$ on 20 000 random points of the workspace.', 7)
        scene_image(writer, fig, os.path.join(FIG, 'viapoint3d_gamma_convergence.png'),
                    '2c · Safety and convergence (3D)',
                    r'$\Gamma(x(t))\geq 1$ for every run (never inside the obstacle + 15 % margin) and $\|x(t)-x^*\|\to 0$.', 7)
        runs, ctrl = scene_robot(writer, fig, ds, obs, starts, Data)
        scene_image(writer, fig, os.path.join(FIG, 'robot_joints_gamma.png'),
                    '3b · Robot joints stay within the real Panda limits',
                    'Normalised joint positions (0/1 = limits), joint velocities over their limits, and Γ of the end-effector for the 5 runs.', 7)
        scene_card(writer, fig, [
            ('Summary', 22),
            ('• Stable LPV-DS learned from demonstrations (P-QLF, $\\dot V<0$ everywhere)', 14),
            ('• 3D modulation $M(x)=E D E^{-1}$: no run enters the obstacle ($\\Gamma_{min}\\geq 1.02$)', 14),
            ('• All 5 trajectories converge to the target (final error < 5 mm)', 14),
            ('• Panda end-effector follows $\\dot x_{mod}$ via $J^{+}(q)$ within joint position and velocity limits', 14)],
            9)
    print('Video saved:', out)


if __name__ == '__main__':
    main()
