"""
MAIN SCRIPT (3D) — reproduces the 3D simulations of the report.

  1. Loads the LPV-DS learned from the 3D demonstrations (train_lpvds.py).
  2. Empirical Lyapunov check: V(x) = (x-x*)^T P (x-x*) and dV/dt on random points.
  3. Defines a 3D ellipsoidal obstacle on the demonstrated path.
  4. Integrates xdot = M(x) f(x) from >= 5 initial points (above, below,
     left, right of the obstacle + a demonstration start).
  5. Reports min Gamma (>= 1 -> no collision), final error and time, and
     saves the figures used in the report.

Usage (env DS_obstacle_avoidance active, from ProyectoFinal):
    python main_3d.py                 # figures in figures/
    python main_3d.py --show          # also opens interactive 3D windows
    python main_3d.py --animate       # live 3D animation (for the video)
"""
import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ds_learning.lpvds import LPVDS
from ds_learning.obstacle3d import Ellipsoid, rotation_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')


# --------------------------------------------------------------------------- #
def build_scenario(ds, x_start, frac=0.45):
    """Obstacle placed on the nominal (learned) path, at `frac` of its length,
    so that without modulation the robot would go straight through it."""
    nominal, _ = integrate(ds, x_start, None)
    seg = np.linalg.norm(np.diff(nominal, axis=1), axis=0)
    s = np.concatenate([[0], np.cumsum(seg)])
    center = nominal[:, np.argmin(np.abs(s - frac * s[-1]))]
    obstacle = Ellipsoid(center=center, axes=[0.06, 0.045, 0.05],
                         rotation=rotation_matrix(0.0, 0.0, np.deg2rad(30)),
                         p=1, sf=1.15, rho=1.0)
    return obstacle, nominal


def initial_points(x_start, ds, offset=0.035):
    """Initial points around the mean demonstration start: centre, above,
    below, left and right (lateral = horizontal, perpendicular to the
    initial direction of motion)."""
    u = ds.evaluate(x_start); u /= np.linalg.norm(u)
    z = np.array([0, 0, 1.0])
    lat = np.cross(z, u); lat /= np.linalg.norm(lat)
    return {'centre': x_start,
            'above':  x_start + offset * z,
            'below':  x_start - offset * z,
            'left':   x_start + offset * lat,
            'right':  x_start - offset * lat}


def integrate(ds, x0, obstacle=None, dt=0.005, T=30.0, tol=5e-3):
    """RK4 integration of xdot = M(x) f(x) (or f(x) if obstacle is None)."""
    def rhs(x):
        f = ds.evaluate(x)
        return f if obstacle is None else obstacle.modulate(x, f)
    x = np.array(x0, float)
    traj = [x.copy()]
    for _ in range(int(T / dt)):
        k1 = rhs(x); k2 = rhs(x + .5 * dt * k1); k3 = rhs(x + .5 * dt * k2); k4 = rhs(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.copy())
        if np.linalg.norm(x - ds.attractor) < tol:
            break
    traj = np.array(traj).T
    return traj, np.arange(traj.shape[1]) * dt


def set_equal_3d(ax, pts):
    lo, hi = pts.min(axis=1), pts.max(axis=1)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


# --------------------------------------------------------------------------- #
def lyapunov_check(ds, Data, n=20000, seed=0):
    """Empirical stability check on random points of a box around the workspace."""
    rng = np.random.default_rng(seed)
    X = Data[:3]
    lo, hi = X.min(axis=1) - 0.15, X.max(axis=1) + 0.15
    S = rng.uniform(lo, hi, size=(n, 3)).T
    V, dV = ds.lyapunov(S)
    ok = np.linalg.norm(S - ds.attractor[:, None], axis=0) > 1e-3
    print(f'[Lyapunov] P eigenvalues: {np.round(np.linalg.eigvalsh(ds.P), 3)} (> 0)')
    print(f'[Lyapunov] max eig(A_k^T P + P A_k) = {ds.stability_report().max():.3f} (< 0)')
    print(f'[Lyapunov] {n} random points: max dV/dt = {dV[ok].max():.2e}  '
          f'-> dV/dt < 0 in {100 * np.mean(dV[ok] < 0):.2f} % of them')
    return S, V, dV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='viapoint3d')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--animate', action='store_true')
    args = ap.parse_args()
    if not (args.show or args.animate):
        matplotlib.use('Agg')
    os.makedirs(FIG, exist_ok=True)

    ds = LPVDS.load(os.path.join(HERE, 'models', args.model + '.json'))
    demo = np.load(os.path.join(HERE, 'models', args.model + '_data.npz'))
    Data, x0_all = demo['Data'], demo['x0_all']
    att = ds.attractor

    # ---- 1) empirical Lyapunov stability ----------------------------------
    S, V, dV = lyapunov_check(ds, Data)

    # ---- 2) scenario ---------------------------------------------------------
    # start of the demonstration closest to the mean start (lies on a real demo)
    mean0 = x0_all.mean(axis=1)
    x_start = x0_all[:, np.argmin(np.linalg.norm(x0_all - mean0[:, None], axis=0))]
    obs, nominal = build_scenario(ds, x_start)
    starts = initial_points(x_start, ds)
    print(f'\nObstacle: centre={np.round(obs.c, 3)}, axes={obs.a}, safety factor={obs.sf}')

    # ---- 3) simulations -------------------------------------------------------
    results = {}
    print(f'\n{"start":>8} | {"min Gamma":>9} | {"min dist [mm]":>13} | {"final err [mm]":>14} | {"time [s]":>8}')
    for name, x0 in starts.items():
        tr, t = integrate(ds, x0, obs)
        free, _ = integrate(ds, x0, None)
        G = np.array([obs.gamma(p) for p in tr.T])          # with safety margin
        G_real = np.array([obs.gamma(p, with_margin=False) for p in tr.T])
        # approx. distance to the real surface along the radial direction
        d_surf = np.array([np.linalg.norm(p - obs.c) * (1 - 1 / np.sqrt(g)) for p, g in zip(tr.T, G_real)])
        err = np.linalg.norm(tr[:, -1] - att)
        results[name] = dict(tr=tr, t=t, free=free, G=G, err=err)
        print(f'{name:>8} | {G.min():9.3f} | {1000 * d_surf.min():13.1f} | {1000 * err:14.2f} | {t[-1]:8.2f}')

    # ---- 4) figures -----------------------------------------------------------
    colors = plt.cm.tab10(np.arange(len(starts)))
    Xs, Ys, Zs = obs.surface()
    Xm, Ym, Zm = obs.surface(with_margin=True)

    allp = np.hstack([r['tr'] for r in results.values()])
    fig = plt.figure(figsize=(16, 7.5))
    for i, zoom in enumerate([False, True]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.plot_surface(Xs, Ys, Zs, color=(0.69, 0.49, 0.49), alpha=0.8, linewidth=0)
        ax.plot_wireframe(Xm, Ym, Zm, color='grey', alpha=0.12, rstride=4, cstride=4)
        if not zoom:
            ax.scatter(*Data[:3, ::2], s=2, c='r', alpha=0.3, label='demonstrations')
        def clip(tr):   # hide the parts outside the zoom box (matplotlib 3D does not clip)
            if not zoom:
                return tr
            tr = tr.copy()
            tr[:, np.any(np.abs(tr - obs.c[:, None]) > 0.13, axis=0)] = np.nan
            return tr
        for (name, r), col in zip(results.items(), colors):
            ax.plot(*clip(r['free']), '--', color=col, lw=1, alpha=0.7)
            ax.plot(*clip(r['tr']), color=col, lw=2.2, label=name)
            if not zoom:
                ax.scatter(*r['tr'][:, 0], color=col, s=30, edgecolor='k')
        ax.scatter(*att, marker='*', s=300, c='k', label='attractor $x^*$')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
        if zoom:
            box = np.column_stack([obs.c - 0.13, obs.c + 0.13])
            set_equal_3d(ax, box)
            ax.view_init(elev=15, azim=-35)
            ax.set_title('Zoom around the obstacle')
        else:
            set_equal_3d(ax, np.hstack([allp, Data[:3]]))
            ax.view_init(elev=22, azim=-60)
            ax.set_title('3D LPV-DS modulated around an ellipsoid\n'
                         '(solid: $M(x)f(x)$, dashed: $f(x)$ without obstacle)')
            ax.legend(loc='upper left', fontsize=8)
    fig.savefig(os.path.join(FIG, f'{args.model}_3d_trajectories.png'), dpi=150, bbox_inches='tight')

    # Gamma(t) and distance to the attractor (convergence)
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for (name, r), col in zip(results.items(), colors):
        a1.plot(r['t'], r['G'], color=col, label=name)
        a2.semilogy(r['t'], np.linalg.norm(r['tr'] - att[:, None], axis=0), color=col, label=name)
    a1.axhline(1, color='k', ls='--', lw=1); a1.set_ylim(0, 15)
    a1.set_xlabel('t [s]'); a1.set_ylabel(r'$\Gamma(x(t))$'); a1.set_title(r'Obstacle distance function ($\Gamma \geq 1$: no collision)')
    a2.set_xlabel('t [s]'); a2.set_ylabel(r'$\|x(t)-x^*\|$ [m]'); a2.set_title('Convergence to the attractor')
    a1.legend(); a2.grid(True, which='both', alpha=0.3)
    fig2.savefig(os.path.join(FIG, f'{args.model}_gamma_convergence.png'), dpi=150, bbox_inches='tight')

    # Empirical Lyapunov plot: V and dV/dt along the trajectories + random samples
    fig3, (b1, b2) = plt.subplots(1, 2, figsize=(12, 4))
    for (name, r), col in zip(results.items(), colors):
        Vf, _ = ds.lyapunov(r['free'])
        tf = np.arange(r['free'].shape[1]) * 0.005
        b1.semilogy(tf, Vf, color=col, label=name)
    b1.set_xlabel('t [s]'); b1.set_ylabel(r'$V(x(t))$'); b1.set_title(r'$V=(x-x^*)^T P (x-x^*)$ along $\dot x=f(x)$')
    b1.grid(True, which='both', alpha=0.3); b1.legend()
    b2.scatter(np.sqrt(V), dV, s=1, alpha=0.3)
    b2.axhline(0, color='k', lw=1)
    b2.set_xlabel(r'$\sqrt{V(x)}$'); b2.set_ylabel(r'$\dot V(x)$')
    b2.set_title(r'$\dot V(x) < 0$ on 20000 random points')
    fig3.savefig(os.path.join(FIG, f'{args.model}_lyapunov.png'), dpi=150, bbox_inches='tight')

    # Modulation field on a slice through the obstacle centre (z = c_z)
    fig4, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    c = obs.c
    gx = np.linspace(c[0] - 0.25, c[0] + 0.25, 45)
    gy = np.linspace(c[1] - 0.25, c[1] + 0.25, 45)
    GX, GY = np.meshgrid(gx, gy)
    for axk, mod in zip(axs, [False, True]):
        U = np.zeros_like(GX); W = np.zeros_like(GX); inside = np.zeros_like(GX, bool)
        for i in range(GX.shape[0]):
            for j in range(GX.shape[1]):
                x = np.array([GX[i, j], GY[i, j], c[2]])
                f = ds.evaluate(x)
                v = obs.modulate(x, f) if mod else f
                U[i, j], W[i, j] = v[0], v[1]
                inside[i, j] = obs.gamma(x, with_margin=False) < 1
        U[inside] = np.nan; W[inside] = np.nan
        axk.streamplot(GX, GY, U, W, color=[0.35, 0.35, 0.8], density=1.3)
        th = np.linspace(0, 2 * np.pi, 200)
        # section of the ellipsoid at z = c_z
        ring = np.array([obs.a[0] * np.cos(th), obs.a[1] * np.sin(th), 0 * th])
        ring = obs.R @ ring + c[:, None]
        axk.fill(ring[0], ring[1], color=(0.69, 0.49, 0.49))
        axk.plot(*att[:2], 'k*', ms=14)
        axk.set_xlim(gx[0], gx[-1]); axk.set_ylim(gy[0], gy[-1])
        axk.set_aspect('equal'); axk.set_xlabel('x [m]'); axk.set_ylabel('y [m]')
        axk.set_title(('Modulated  $M(x)f(x)$' if mod else 'Learned LPV-DS  $f(x)$') + f'  (slice z = {c[2]:.2f} m)')
    fig4.savefig(os.path.join(FIG, f'{args.model}_slice_field.png'), dpi=150, bbox_inches='tight')
    print('\nFigures saved in', FIG)

    # ---- 5) optional animation ------------------------------------------------
    if args.animate:
        from matplotlib.animation import FuncAnimation
        figA = plt.figure(figsize=(9, 8)); axA = figA.add_subplot(111, projection='3d')
        axA.plot_surface(Xs, Ys, Zs, color=(0.69, 0.49, 0.49), alpha=0.85, linewidth=0)
        axA.scatter(*att, marker='*', s=300, c='k')
        set_equal_3d(axA, allp); axA.view_init(elev=22, azim=-60)
        axA.set_xlabel('x [m]'); axA.set_ylabel('y [m]'); axA.set_zlabel('z [m]')
        lines = [axA.plot([], [], [], color=col, lw=2)[0] for col in colors]
        dots = [axA.plot([], [], [], 'o', color=col, ms=8)[0] for col in colors]
        n_max = max(r['tr'].shape[1] for r in results.values())
        step = 4
        def update(k):
            i = k * step
            for ln, dt_, r in zip(lines, dots, results.values()):
                j = min(i, r['tr'].shape[1] - 1)
                ln.set_data(r['tr'][0, :j + 1], r['tr'][1, :j + 1]); ln.set_3d_properties(r['tr'][2, :j + 1])
                dt_.set_data([r['tr'][0, j]], [r['tr'][1, j]]); dt_.set_3d_properties([r['tr'][2, j]])
            axA.set_title(f't = {i * 0.005:.2f} s')
            return lines + dots
        anim = FuncAnimation(figA, update, frames=n_max // step + 1, interval=20, blit=False)
    if args.show or args.animate:
        plt.show()


if __name__ == '__main__':
    main()
