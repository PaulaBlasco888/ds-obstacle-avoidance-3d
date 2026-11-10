"""
Lab Session 2 / roadmap step 3: "Unify both parts in 2D".

Replaces the default linear attractor of the obstacle-avoidance package
(exercise2_obstacle) with the LPV-DS learned from demonstrations
(train_lpvds.py), and checks that the modulated trajectories
    xdot = M(x) f_LPV(x),   M(x) = E(x) D(x) E(x)^-1
  (1) follow the shape of the learned DS,
  (2) avoid the obstacle,
  (3) still converge to the attractor.

Usage (env DS_obstacle_avoidance active, from the ProyectoFinal folder):
    python train_lpvds.py                    # (only once) -> models/sshape2d.json
    python session2_lpvds_obstacle_2d.py     # figures in figures/
    python session2_lpvds_obstacle_2d.py --animate   # live animation (package GUI)
"""
import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import (
    obs_avoidance_interpolation_moving)
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import linearAttractor
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields

from ds_learning.lpvds import LPVDS

HERE = os.path.dirname(os.path.abspath(__file__))

# Scenario per model: (obstacle centre, semi-axes, x-range, y-range)
SCENARIOS = {
    'sshape2d': dict(x0=[-3.0, 0.68], a=[0.28, 0.18], xr=[-5.0, -1.0], yr=[-0.8, 2.2]),
    'lshape2d': dict(x0=[-3.0, -0.1], a=[0.25, 0.2], xr=[-4.8, -1.2], yr=[-0.8, 2.4]),
    'ashape2d': dict(x0=[-2.6, 1.0], a=[0.3, 0.25], xr=[-5.0, 0.2], yr=[-0.8, 2.4]),
}


def make_obstacles(sc):
    # Ellipse (p=1) with a 20 % safety margin (sf)
    return [Obstacle(a=sc['a'], p=[1, 1], x0=sc['x0'], th_r=0, sf=1.2)]


def gamma_obstacle(x, obs):
    """Gamma(x) of an (axis-aligned) ellipse, including the safety factor."""
    a = np.array(obs.sf) * np.array(obs.a)
    return np.sum(((x - np.array(obs.x0)) / a) ** 2)


def integrate(ds, x0, obs, dt=0.002, n_max=6000, tol=2e-2):
    """RK4 integration of xdot = M(x) f(x), keeping the learned speed profile."""
    def rhs(x):
        return obs_avoidance_interpolation_moving(x, ds(x), obs)
    x = np.array(x0, float); traj = [x.copy()]
    for _ in range(n_max):
        k1 = rhs(x); k2 = rhs(x + .5 * dt * k1); k3 = rhs(x + .5 * dt * k2); k4 = rhs(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.copy())
        if np.linalg.norm(x - ds.attractor) < tol:
            break
    return np.array(traj).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sshape2d')
    ap.add_argument('--animate', action='store_true')
    ap.add_argument('--grid', type=int, default=60)
    args = ap.parse_args()

    ds = LPVDS.load(os.path.join(HERE, 'models', args.model + '.json'))
    demo = np.load(os.path.join(HERE, 'models', args.model + '_data.npz'))
    att, x0_all, Data = ds.attractor, demo['x0_all'], demo['Data']
    sc = SCENARIOS[args.model]
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Figure 1: linear attractor (original code) vs learned LPV-DS        #
    # both modulated around the same obstacle                             #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, (title, dsys) in zip(axes, [('Linear attractor  f(x) = -(x - x*)', linearAttractor),
                                       ('Learned LPV-DS  f(x) = sum gamma_k (A_k x + b_k)', ds)]):
        plt.sca(ax)   # the package draws obstacles on plt.gca()
        Simulation_vectorFields(sc['xr'], sc['yr'], args.grid, make_obstacles(sc),
                                xAttractor=att, dynamicalSystem=dsys, noTicks=False,
                                figHandle=[fig, ax], streamColor=[0.55, 0.55, 0.85])
        ax.set_title('Modulated: ' + title, fontsize=12)
    axes[1].plot(Data[0], Data[1], 'r.', ms=1.5, alpha=0.6, label='demonstrations')
    axes[1].legend(loc='lower right')
    fig.savefig(os.path.join(HERE, 'figures', f'{args.model}_linear_vs_lpvds.png'), dpi=150, bbox_inches='tight')

    # ------------------------------------------------------------------ #
    # Figure 2: trajectories without / with obstacle + metrics            #
    # ------------------------------------------------------------------ #
    obs = make_obstacles(sc)
    for o in obs:
        o.draw_ellipsoid(numPoints=60)
    starts = list(x0_all.T)
    rng = np.random.default_rng(3)
    starts += [x + rng.normal(0, 0.12, 2) for x in x0_all.T]   # perturbed starts

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    ax2.add_patch(plt.Polygon(obs[0].x_obs_sf, color=[0.9, 0.9, 0.9], zorder=1))
    ax2.add_patch(plt.Polygon(obs[0].x_obs, color=np.array([176, 124, 124]) / 255, zorder=2))
    ax2.plot(Data[0], Data[1], 'r.', ms=1.5, alpha=0.4, label='demonstrations')

    print(f'{"start":>22} | {"min Gamma":>9} | {"final error":>11} | steps')
    for i, x0 in enumerate(starts):
        free = ds.simulate(x0, dt=0.002, n_steps=6000, tol=2e-2)            # no obstacle
        tr = integrate(ds, x0, obs)                                          # modulated
        g = min(gamma_obstacle(p, obs[0]) for p in tr.T)
        err = np.linalg.norm(tr[:, -1] - att)
        print(f'{np.round(x0, 2)!s:>22} | {g:9.3f} | {err:11.4f} | {tr.shape[1]}')
        ax2.plot(free[0], free[1], 'k--', lw=1, alpha=0.5, label='LPV-DS (no obstacle)' if i == 0 else None)
        ax2.plot(tr[0], tr[1], lw=2.2, label='LPV-DS + modulation' if i == 0 else None)
        ax2.plot(*x0, 'ko', ms=5)
    ax2.plot(*att, 'k*', ms=18, label='attractor')
    ax2.set_xlim(sc['xr']); ax2.set_ylim(sc['yr']); ax2.set_aspect('equal')
    ax2.set_title('Learned LPV-DS modulated around an obstacle (Gamma >= 1 everywhere)')
    ax2.legend(loc='upper right', fontsize=9)
    fig2.savefig(os.path.join(HERE, 'figures', f'{args.model}_trajectories_obstacle.png'), dpi=150, bbox_inches='tight')
    print('Figures saved in', os.path.join(HERE, 'figures'))

    # ------------------------------------------------------------------ #
    # Optional: live animation with the package (now using our DS)        #
    # ------------------------------------------------------------------ #
    if args.animate:
        from dynamic_obstacle_avoidance.visualization.animated_simulation import run_animation
        run_animation(np.array(starts).T, obs=make_obstacles(sc), x_range=sc['xr'], y_range=sc['yr'],
                      dt=0.003, N_simuMax=2000, convergenceMargin=0.02, sleepPeriod=0.001,
                      attractorPos=att, RK4_int=True, dynamicalSystem=ds, hide_ticks=False)
    elif matplotlib.get_backend().lower() != 'agg':
        plt.show()


if __name__ == '__main__':
    main()
