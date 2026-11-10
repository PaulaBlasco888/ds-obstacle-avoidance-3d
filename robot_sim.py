"""
Robotic manipulator control with the modulated 3D DS (Franka Emika Panda,
Robotics Toolbox for Python, kinematic simulation).

At every control step (dt = 10 ms):
    1. read the end-effector position p = FK(q)
    2. evaluate the modulated DS in the task frame:  v = M(x) f(x),  x = p - t_task
    3. keep the gripper pointing down: w = K_o * orientation error
    4. joint velocities with damped least squares + null-space joint centring:
           qdot = J^+ [v; w] + (I - J^+ J) qdot_0
    5. saturate with the real Panda joint velocity limits and integrate q.

The DS is translation-equivariant, so the learned task is simply moved into
the Panda workspace with the offset t_task (demonstrations were recorded with
another robot, in another base frame).

Usage (from ProyectoFinal, env active):
    python robot_sim.py               # 5 runs + figures
    python robot_sim.py --show        # interactive animation of one run
    python robot_sim.py --save-video  # MP4 of the robot (figures/robot_*.mp4)
"""
import argparse
import os
import numpy as np
import matplotlib

import roboticstoolbox as rtb
from spatialmath import SE3

from ds_learning.lpvds import LPVDS
from main_3d import build_scenario, initial_points, integrate, set_equal_3d

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')

# Real limits of the Franka Emika Panda (datasheet)
QDOT_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])   # rad/s
EE_V_MAX = 1.7                                                        # m/s (Cartesian)


class PandaDSController:
    def __init__(self, ds, obstacle, t_task, dt=0.01, k_ori=2.0, damping=0.02, k_null=0.5):
        self.robot = rtb.models.DH.Panda()
        self.ds, self.obs, self.t_task = ds, obstacle, np.asarray(t_task, float)
        self.dt, self.k_ori, self.lam, self.k_null = dt, k_ori, damping, k_null
        self.qlim = self.robot.qlim
        self.q_mid = self.qlim.mean(axis=0)
        self.R_des = self.robot.fkine(self.robot.qr).R          # gripper pointing down

    # ---- task-frame helpers -------------------------------------------------
    def to_task(self, p):
        p = np.asarray(p, float)
        return p - (self.t_task if p.ndim == 1 else self.t_task[:, None])

    def to_robot(self, x):
        x = np.asarray(x, float)
        return x + (self.t_task if x.ndim == 1 else self.t_task[:, None])

    def desired_velocity(self, p):
        x = self.to_task(p)
        f = self.ds.evaluate(x)
        return self.obs.modulate(x, f) if self.obs is not None else f

    # ---- inverse kinematics for the initial configuration ------------------
    def initial_q(self, p0):
        T = SE3.Rt(self.R_des, p0)
        best = None
        for seed in range(10):
            q0 = self.robot.qr if seed == 0 else np.random.default_rng(seed).uniform(*self.qlim)
            sol = self.robot.ikine_LM(T, q0=q0, joint_limits=True)
            if sol.success:
                margin = np.min(np.minimum(sol.q - self.qlim[0], self.qlim[1] - sol.q))
                if best is None or margin > best[0]:
                    best = (margin, sol.q)
        if best is None:
            raise RuntimeError(f'No IK solution for {p0}')
        return best[1]

    # ---- one control step -------------------------------------------------------
    def step(self, q):
        T = self.robot.fkine(q)
        p, R = T.t, T.R
        v = self.desired_velocity(p)
        n = np.linalg.norm(v)
        if n > EE_V_MAX:
            v = v / n * EE_V_MAX
        # orientation error (axis-angle of R_des R^T), as angular velocity command
        Re = self.R_des @ R.T
        ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        if ang < 1e-9:
            w = np.zeros(3)
        else:
            axis = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * np.sin(ang))
            w = self.k_ori * ang * axis
        J = self.robot.jacob0(q)                                  # 6x7 geometric Jacobian
        # damped least-squares pseudo-inverse  J^T (J J^T + lambda^2 I)^-1
        J_pinv = J.T @ np.linalg.inv(J @ J.T + self.lam ** 2 * np.eye(6))
        # secondary task: stay away from joint limits (null space)
        qd0 = -self.k_null * (q - self.q_mid) / ((self.qlim[1] - self.qlim[0]) / 2) ** 2
        qd = J_pinv @ np.hstack([v, w]) + (np.eye(7) - J_pinv @ J) @ qd0
        # joint velocity saturation (keeps the direction of qd)
        scale = np.max(np.abs(qd) / QDOT_MAX)
        if scale > 1:
            qd = qd / scale
        q_new = np.clip(q + qd * self.dt, self.qlim[0], self.qlim[1])
        self.last_track_err = np.linalg.norm(J[:3] @ qd - v)    # realised vs commanded EE velocity
        return q_new, qd, p, v

    def run(self, p0, T=25.0, tol=5e-3):
        q = self.initial_q(p0)
        log = {k: [] for k in ('q', 'qd', 'p', 'v', 'gamma', 'track')}
        target = self.to_robot(self.ds.attractor)
        for _ in range(int(T / self.dt)):
            q_new, qd, p, v = self.step(q)
            log['q'].append(q); log['qd'].append(qd); log['p'].append(p); log['v'].append(v)
            log['track'].append(self.last_track_err)
            log['gamma'].append(self.obs.gamma(self.to_task(p)) if self.obs is not None else np.inf)
            q = q_new
            if np.linalg.norm(p - target) < tol:
                break
        out = {k: np.array(val) for k, val in log.items()}
        out['t'] = np.arange(len(out['q'])) * self.dt
        return out

    def joint_positions(self, q):
        """3D positions of the joints (for drawing the arm)."""
        Ts = self.robot.fkine_all(q)
        P = [T.t for T in Ts]
        P.append(self.robot.fkine(q).t)          # include the tool (gripper tip)
        return np.array(P)


# --------------------------------------------------------------------------- #
def draw_arm(ax, ctrl, q, color='0.25', alpha=1.0, lw=6):
    P = ctrl.joint_positions(q)
    h1, = ax.plot(P[:, 0], P[:, 1], P[:, 2], '-', color=color, lw=lw, alpha=alpha, solid_capstyle='round')
    h2, = ax.plot(P[:, 0], P[:, 1], P[:, 2], 'o', color='k', ms=4, alpha=alpha)
    return [h1, h2]


def draw_scene(ax, ctrl):
    obs = ctrl.obs
    X, Y, Z = obs.surface()
    ax.plot_surface(X + ctrl.t_task[0], Y + ctrl.t_task[1], Z + ctrl.t_task[2],
                    color=(0.69, 0.49, 0.49), alpha=0.8, linewidth=0)
    att = ctrl.to_robot(ctrl.ds.attractor)
    ax.scatter(*att, marker='*', s=250, c='k')
    # floor / base
    ax.plot([0], [0], [0], 's', color='k', ms=10)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='viapoint3d')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--save-video', action='store_true')
    args = ap.parse_args()
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)

    ds = LPVDS.load(os.path.join(HERE, 'models', args.model + '.json'))
    demo = np.load(os.path.join(HERE, 'models', args.model + '_data.npz'))
    Data, x0_all = demo['Data'], demo['x0_all']
    mean0 = x0_all.mean(axis=1)
    x_start = x0_all[:, np.argmin(np.linalg.norm(x0_all - mean0[:, None], axis=0))]
    obs, _ = build_scenario(ds, x_start)
    starts = initial_points(x_start, ds)

    # place the task in the Panda workspace: centre of the demonstrations -> (0.5, 0, 0.4)
    t_task = np.array([0.50, 0.0, 0.40]) - Data[:3].mean(axis=1)
    ctrl = PandaDSController(ds, obs, t_task)
    print('Task offset t_task =', np.round(t_task, 3))

    runs = {}
    print(f'\n{"start":>8} | {"min Gamma":>9} | {"final err [mm]":>14} | {"max |Jqd - v| [mm/s]":>20} | '
          f'{"max |qd|/qd_max":>15} | {"min joint margin [deg]":>22} | time [s]')
    for name, x0 in starts.items():
        r = ctrl.run(ctrl.to_robot(x0))
        margin = np.min(np.minimum(r['q'] - ctrl.qlim[0], ctrl.qlim[1] - r['q']))
        err = np.linalg.norm(r['p'][-1] - ctrl.to_robot(ds.attractor))
        runs[name] = r
        print(f'{name:>8} | {r["gamma"].min():9.3f} | {1000 * err:14.2f} | {1000 * r["track"].max():20.2f} | '
              f'{np.max(np.abs(r["qd"]) / QDOT_MAX):15.2f} | {np.rad2deg(margin):22.1f} | {r["t"][-1]:.2f}')

    colors = plt.cm.tab10(np.arange(len(runs)))

    # ---- Figure: robot + end-effector paths ------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    draw_scene(ax, ctrl)
    r0 = runs['centre']
    for k in np.linspace(0, len(r0['q']) - 1, 5).astype(int):
        draw_arm(ax, ctrl, r0['q'][k], alpha=0.25 + 0.75 * k / len(r0['q']))
    for (name, r), col in zip(runs.items(), colors):
        ax.plot(*r['p'].T, color=col, lw=2, label=f'end-effector ({name})')
    ax.scatter(*ctrl.to_robot(Data[:3, ::4]), s=1, c='r', alpha=0.2)
    set_equal_3d(ax, np.vstack([np.vstack([r['p'] for r in runs.values()]), [[0, 0, 0]]]).T)
    ax.view_init(elev=20, azim=-50)
    ax.set_title('Franka Emika Panda following the modulated LPV-DS\n'
                 r'$\dot q = J^{+}(q)\,[\dot x_{mod};\ \omega]$')
    ax.legend(loc='upper left', fontsize=8)
    fig.savefig(os.path.join(FIG, 'robot_panda_3d.png'), dpi=150, bbox_inches='tight')

    # ---- Figure: joints vs limits, joint velocities vs limits, Gamma -------------
    fig2, axs = plt.subplots(1, 3, figsize=(17, 4.2))
    r = runs['centre']
    qn = (r['q'] - ctrl.qlim[0]) / (ctrl.qlim[1] - ctrl.qlim[0])
    for j in range(7):
        axs[0].plot(r['t'], qn[:, j], label=f'q{j + 1}')
        axs[1].plot(r['t'], r['qd'][:, j] / QDOT_MAX[j], label=f'q{j + 1}')
    for a in axs[:2]:
        a.axhline(1, color='k', ls='--', lw=1); a.axhline(0 if a is axs[0] else -1, color='k', ls='--', lw=1)
        a.set_xlabel('t [s]')
    axs[0].set_title('Joint positions (normalised: 0 = lower, 1 = upper limit)')
    axs[1].set_title(r'Joint velocities $\dot q_i/\dot q_{i,max}$')
    axs[0].legend(ncol=4, fontsize=8)
    for (name, r), col in zip(runs.items(), colors):
        axs[2].plot(r['t'], r['gamma'], color=col, label=name)
    axs[2].axhline(1, color='k', ls='--', lw=1); axs[2].set_ylim(0, 15)
    axs[2].set_xlabel('t [s]'); axs[2].set_title(r'$\Gamma$ of the end-effector'); axs[2].legend(fontsize=8)
    fig2.savefig(os.path.join(FIG, 'robot_joints_gamma.png'), dpi=150, bbox_inches='tight')
    print('\nFigures saved in', FIG)

    # ---- Animation (interactive or MP4) ----------------------------------------
    if args.show or args.save_video:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        figA = plt.figure(figsize=(9, 7.5))
        axA = figA.add_subplot(111, projection='3d')
        draw_scene(axA, ctrl)
        set_equal_3d(axA, np.vstack([np.vstack([r['p'] for r in runs.values()]), [[0, 0, 0]]]).T)
        axA.view_init(elev=20, azim=-50)
        sel = ['centre', 'above', 'right']
        artists = []
        trails = {n: axA.plot([], [], [], color=colors[list(runs).index(n)], lw=2, label=n)[0] for n in sel}
        axA.legend(loc='upper left')
        step = 4
        n_frames = max(len(runs[n]['q']) for n in sel) // step + 1

        def update(k):
            for h in artists:
                h.remove()
            artists.clear()
            i = k * step
            for n in sel:
                rr = runs[n]
                j = min(i, len(rr['q']) - 1)
                trails[n].set_data(rr['p'][:j + 1, 0], rr['p'][:j + 1, 1])
                trails[n].set_3d_properties(rr['p'][:j + 1, 2])
                artists.extend(draw_arm(axA, ctrl, rr['q'][j], color=colors[list(runs).index(n)] * 0.6, lw=5))
            axA.set_title(f'Panda + modulated LPV-DS    t = {i * ctrl.dt:.2f} s')
            return artists
        anim = FuncAnimation(figA, update, frames=n_frames, interval=40)
        if args.save_video:
            import imageio_ffmpeg
            plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
            out = os.path.join(FIG, 'robot_panda.mp4')
            anim.save(out, writer=FFMpegWriter(fps=25, bitrate=3000), dpi=110)
            print('Video saved:', out)
        if args.show:
            plt.show()


if __name__ == '__main__':
    main()
