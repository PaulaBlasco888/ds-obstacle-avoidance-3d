"""
Step 1 of the project roadmap: learn a stable non-linear DS (LPV-DS) from the
Lab Session 1 demonstrations and save it to models/<name>.json.

Usage (from the ProyectoFinal folder, with the conda env active):
    python train_lpvds.py                   # 2D S-shape (Lab Session 2 work)
    python train_lpvds.py --dataset 3D_viapoint_3 --name viapoint3d --nb_traj 6 --eps 0.01 --direction_weight 4
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from ds_learning.lpvds import LPVDS, load_ds_opt_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, '..', 'Lab Session 1 Code-20261109', 'exercise1_learning',
                        'exercise1_learning', 'ds-opt', 'datasets')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='2D_Sshape', help='file in ds-opt/datasets (without .mat)')
    ap.add_argument('--name', default='sshape2d', help='output model name')
    ap.add_argument('--K', type=int, default=None, help='number of Gaussians (default: BIC)')
    ap.add_argument('--sub_sample', type=int, default=None)
    ap.add_argument('--nb_traj', type=int, default=None)
    ap.add_argument('--lyapunov', default='pqlf', choices=['qlf', 'pqlf'])
    ap.add_argument('--eps', type=float, default=0.1, help='stability margin: A_k^T P + P A_k <= -eps I')
    ap.add_argument('--direction_weight', type=float, default=0.0,
                    help='>0: cluster on position + velocity direction (PC-GMM-like)')
    args = ap.parse_args()

    is3d = args.dataset.startswith('3D')
    sub = args.sub_sample or (3 if is3d else 1)
    Data, att, x0_all = load_ds_opt_dataset(os.path.join(DATASETS, args.dataset + '.mat'),
                                            sub_sample=sub, nb_trajectories=args.nb_traj)
    M = Data.shape[0] // 2
    X, Xdot = Data[:M], Data[M:]
    print(f'Dataset {args.dataset}: dim={M}, {X.shape[1]} samples, '
          f'{x0_all.shape[1]} demos, attractor={np.round(att, 3)}')

    ds = LPVDS().fit(X, Xdot, att, K=args.K, lyapunov=args.lyapunov, eps=args.eps,
                     direction_weight=args.direction_weight, verbose=True)

    # --- metrics -----------------------------------------------------------
    Xdot_hat = ds.evaluate(X)
    rmse = np.mean(np.linalg.norm(Xdot_hat - Xdot, axis=0))
    print(f'Mean velocity error on training data: {rmse:.4f}')
    print('max eig(A_k^T P + P A_k) per component (must all be < 0):',
          np.round(ds.stability_report(), 4))

    os.makedirs(os.path.join(HERE, 'models'), exist_ok=True)
    out = os.path.join(HERE, 'models', args.name + '.json')
    ds.save(out)
    np.savez(os.path.join(HERE, 'models', args.name + '_data.npz'), Data=Data, att=att, x0_all=x0_all)
    print('Saved', out)

    # --- quick plot of reproductions (2D only) --------------------------------
    if M == 2:
        os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
        lo, hi = X.min(axis=1), X.max(axis=1)
        pad = 0.25 * (hi - lo)
        xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 60)
        ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 60)
        XX, YY = np.meshgrid(xs, ys)
        V = ds.evaluate(np.vstack([XX.ravel(), YY.ravel()]))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.streamplot(XX, YY, V[0].reshape(XX.shape), V[1].reshape(XX.shape),
                      color=[0.6, 0.6, 0.8], density=1.4)
        ax.plot(X[0], X[1], 'r.', ms=2, label='demonstrations')
        for x0 in x0_all.T:
            tr = ds.simulate(x0, dt=0.005)
            ax.plot(tr[0], tr[1], 'k', lw=2)
        ax.plot(*att, 'k*', ms=15, label='attractor')
        ax.set_title(f'Learned LPV-DS ({args.dataset}, K={ds.K})')
        ax.set_aspect('equal'); ax.legend()
        fig.savefig(os.path.join(HERE, 'figures', f'{args.name}_lpvds.png'), dpi=150, bbox_inches='tight')
        print('Figure saved in figures/')


if __name__ == '__main__':
    main()
