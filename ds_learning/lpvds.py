"""
GMM-based LPV-DS (Linear Parameter Varying Dynamical System) in Python.

Python port of the LPV-DS learning pipeline seen in Lab Session 1
(ds-opt / phys-gmm, N. Figueroa & A. Billard, LASA-EPFL):

    f(x) = sum_k gamma_k(x) * (A_k x + b_k)

    * gamma_k(x)  : posterior probability of Gaussian k of a GMM fitted on the
                    demonstrated positions (sum_k gamma_k(x) = 1, gamma_k >= 0)
    * A_k, b_k    : found with a convex SDP (constr_type = 0 in ds-opt):
                        A_k^T + A_k  <= -eps * I
                        b_k = -A_k x*
                    which guarantees global asymptotic stability at x* with the
                    quadratic Lyapunov function V(x) = (x - x*)^T (x - x*).

Proof (QLF):  with b_k = -A_k x*  ->  f(x) = sum_k gamma_k A_k (x - x*)
    dV/dt = 2 (x-x*)^T f(x) = sum_k gamma_k (x-x*)^T (A_k + A_k^T) (x-x*) < 0
    for every x != x*, because every gamma_k >= 0, sum gamma_k = 1 and every
    (A_k + A_k^T) is negative definite.

The MATLAB version needs the Statistics Toolbox (fitgmdist) + YALMIP/SeDuMi;
here the GMM is fitted with our own EM implementation (numpy) and the SDP is
solved with cvxpy (Clarabel solver).
"""

import json
import logging
import numpy as np


def _cvxpy():
    """Imports cvxpy. Some optional solvers (SCS, OSQP) may fail to load on
    Windows with Smart App Control; only CLARABEL is used, so those import
    warnings are silenced."""
    logging.disable(logging.WARNING)
    try:
        import cvxpy as cp
    finally:
        logging.disable(logging.NOTSET)
    return cp
from scipy.io import loadmat
from scipy.special import logsumexp


# --------------------------------------------------------------------------- #
#  Data loading (port of load_dataset_DS.m / processDataStructure.m)          #
# --------------------------------------------------------------------------- #
def load_ds_opt_dataset(mat_path, sub_sample=1, nb_trajectories=None, seed=0):
    """Loads a dataset from ds-opt/datasets (e.g. 2D_Sshape.mat, 3D_Cshape_top.mat).

    Returns
    -------
    Data  : (2M, N) stacked [positions; velocities] of all demonstrations
    att   : (M,)    attractor (mean of the last point of every demonstration)
    x0_all: (M, n_traj) initial points of the demonstrations
    """
    raw = loadmat(mat_path)['data']
    trajs = [np.asarray(raw.flat[i], dtype=float) for i in range(raw.size)]
    if nb_trajectories is not None and nb_trajectories < len(trajs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(trajs), nb_trajectories, replace=False)
        trajs = [trajs[i] for i in sorted(idx)]
    trajs = [t[:, ::sub_sample] for t in trajs]

    M = trajs[0].shape[0] // 2
    ends = np.stack([t[:M, -1] for t in trajs], axis=1)
    att = ends.mean(axis=1)
    shifts = ends - att[:, None]

    Data, x0_all = [], []
    for t, s in zip(trajs, shifts.T):
        t = t.copy()
        t[:M] -= s[:, None]                 # all demos end exactly at att
        t[M:, -1] = 0.0                     # zero velocity at the target
        t[M:, -2] = t[M:, -1] / 2
        t[M:, -3] = (t[M:, -3] + t[M:, -2]) / 2
        Data.append(t)
        x0_all.append(t[:M, 0])
    return np.hstack(Data), att, np.stack(x0_all, axis=1)


# --------------------------------------------------------------------------- #
#  Gaussian Mixture Model (EM + BIC), numpy only                              #
#  (replaces MATLAB's fitgmdist used by fit_gmm.m, option type=1)             #
# --------------------------------------------------------------------------- #
def gauss_logpdf(X, mu, S):
    """log N(x | mu, S) for X of shape (N, M)."""
    M = X.shape[1]
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * np.sum(z ** 2, axis=0) - np.log(np.diag(L)).sum() - 0.5 * M * np.log(2 * np.pi)


def _kmeans_pp_init(X, K, rng):
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, K):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centers)


def fit_gmm_em(X, K, n_init=5, max_iter=300, tol=1e-6, reg=1e-6, seed=0):
    """EM for a full-covariance GMM. X: (N, M).
    Returns priors (K,), mu (K,M), sigma (K,M,M), log-likelihood."""
    N, M = X.shape
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        mu = _kmeans_pp_init(X, K, rng)
        sigma = np.tile(np.cov(X.T) / K + reg * np.eye(M), (K, 1, 1))
        priors = np.full(K, 1.0 / K)
        prev = -np.inf
        for _ in range(max_iter):
            # E-step: responsibilities r_nk
            logp = np.stack([np.log(priors[k]) + gauss_logpdf(X, mu[k], sigma[k])
                             for k in range(K)], axis=1)
            ll_n = logsumexp(logp, axis=1)
            r = np.exp(logp - ll_n[:, None])
            ll = ll_n.sum()
            # M-step
            Nk = r.sum(axis=0) + 1e-12
            priors = Nk / N
            mu = (r.T @ X) / Nk[:, None]
            for k in range(K):
                D = X - mu[k]
                sigma[k] = (r[:, k, None] * D).T @ D / Nk[k] + reg * np.eye(M)
            if abs(ll - prev) < tol * abs(ll):
                break
            prev = ll
        if best is None or ll > best[3]:
            best = (priors.copy(), mu.copy(), sigma.copy(), ll)
    return best


def gmm_bic(ll, K, M, N):
    n_params = K * (M + M * (M + 1) / 2) + (K - 1)
    return -2 * ll + n_params * np.log(N)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def adjust_covariances(priors, sigma, tot_scale_fact=1.0, rel_scale_fact=0.2):
    """Port of adjust_Covariances.m: dilates covariance matrices that are too
    thin so that the local linear dynamics have a wider region of influence
    (gives smoother global streamlines)."""
    sigma = sigma.copy()
    K, dim = sigma.shape[0], sigma.shape[1]
    baseline = 0.5 / len(priors)
    for k in range(K):
        L, V = np.linalg.eigh(sigma[k])      # ascending eigenvalues
        if priors[k] > baseline:
            L = tot_scale_fact * L
        ratio = L[-2] / L[-1]
        if ratio < rel_scale_fact:
            L[:-1] = L[:-1] + L[-1] * (rel_scale_fact - ratio)
        sigma[k] = V @ np.diag(L) @ V.T
    return sigma


# --------------------------------------------------------------------------- #
#  LPV-DS                                                                      #
# --------------------------------------------------------------------------- #
class LPVDS:
    def __init__(self, priors=None, mu=None, sigma=None, A=None, b=None, attractor=None, P=None):
        self.P = None if P is None else np.asarray(P, float)
        # shapes: priors (K,), mu (K,M), sigma (K,M,M), A (K,M,M), b (K,M), attractor (M,)
        self.priors = None if priors is None else np.asarray(priors, float)
        self.mu = None if mu is None else np.asarray(mu, float)
        self.sigma = None if sigma is None else np.asarray(sigma, float)
        self.A = None if A is None else np.asarray(A, float)
        self.b = None if b is None else np.asarray(b, float)
        self.attractor = None if attractor is None else np.asarray(attractor, float)

    @property
    def K(self):
        return len(self.priors)

    @property
    def dim(self):
        return self.mu.shape[1]

    # ----------------------------------------------------------- learning --
    @staticmethod
    def learn_P(X, Xdot, attractor, min_eig=0.05):
        """Learns the matrix P of the Lyapunov function V(x) = (x-x*)^T P (x-x*)
        so that the demonstrations are as compatible as possible with dV/dt < 0:
            min_P  sum_n max(0, (x_n-x*)^T P xdot_n / (|x_n-x*| |xdot_n|))
            s.t.   P >= min_eig*I,  trace(P) = M
        This is a convex problem (linear in P). It plays the role of
        learn_wsaqf.m in ds-opt (constr_type = 2, 'P-QLF')."""
        cp = _cvxpy()
        M = X.shape[0]
        Xc = X - np.asarray(attractor).reshape(M, 1)
        nx = np.linalg.norm(Xc, axis=0); nv = np.linalg.norm(Xdot, axis=0)
        ok = (nx > 1e-6) & (nv > 1e-6)
        Xc, Xd = Xc[:, ok] / nx[ok], Xdot[:, ok] / nv[ok]
        P = cp.Variable((M, M), symmetric=True)
        dV = cp.sum(cp.multiply(Xc, P @ Xd), axis=0)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.pos(dV + 0.05))),
                          [P >> min_eig * np.eye(M), cp.trace(P) == M])
        prob.solve(solver=cp.CLARABEL)
        return (P.value + P.value.T) / 2

    def fit(self, X, Xdot, attractor, K=None, max_K=8, rel_dilation=None,
            eps=0.1, seed=0, lyapunov='pqlf', p_min_eig=0.2, direction_weight=0.0,
            verbose=False):
        """Learn the LPV-DS from demonstrations.

        X, Xdot   : (M, N) positions and velocities
        attractor : (M,)   target x*
        K         : number of Gaussians (None -> chosen with BIC up to max_K)
        lyapunov  : 'qlf'  -> V = (x-x*)^T (x-x*),   A_k^T + A_k   < 0
                    'pqlf' -> V = (x-x*)^T P (x-x*), A_k^T P + P A_k < 0
        direction_weight : w > 0 clusters on [x ; w * s * xdot/|xdot|] (s = position
                    std) so that points that are close in space but move in different
                    directions fall in different Gaussians (same idea as the
                    physically-consistent PC-GMM of Lab 1). Only the position part of
                    each Gaussian is kept for gamma_k(x).
        """
        cp = _cvxpy()

        X, Xdot = np.asarray(X, float), np.asarray(Xdot, float)
        M, N = X.shape
        self.attractor = np.asarray(attractor, float).reshape(M)

        # ---- Step 2: GMM (EM + BIC model selection) ----------------------------
        F = X
        if direction_weight > 0:
            nv = np.linalg.norm(Xdot, axis=0, keepdims=True)
            dirs = np.divide(Xdot, nv, out=np.zeros_like(Xdot), where=nv > 1e-9)
            F = np.vstack([X, direction_weight * X.std(axis=1).mean() * dirs])
        Mf = F.shape[0]
        if K is None:
            best = None
            for k in range(1, max_K + 1):
                g = fit_gmm_em(F.T, k, seed=seed)
                bic = gmm_bic(g[3], k, Mf, N)
                if verbose:
                    print(f'  K={k}: BIC={bic:.1f}')
                if best is None or bic < best[0]:
                    best = (bic, g)
            gmm = best[1]
        else:
            gmm = fit_gmm_em(F.T, K, n_init=10, seed=seed)
        # keep only the position marginal of each Gaussian
        self.priors, self.mu, sig = gmm[0], gmm[1][:, :M], gmm[2][:, :M, :M]
        if rel_dilation is None:
            rel_dilation = 0.2 if M == 2 else 0.75
        self.sigma = adjust_covariances(self.priors, sig, 1.0, rel_dilation)
        if verbose:
            print(f'GMM with K={self.K} Gaussians')

        # ---- Step 3: Lyapunov matrix P ----------------------------------------
        self.P = np.eye(M) if lyapunov == 'qlf' else self.learn_P(X, Xdot, self.attractor, p_min_eig)
        if verbose:
            print('Lyapunov matrix P =\n', np.round(self.P, 3))

        # ---- Step 4: convex optimisation of A_k (b_k = -A_k x*) --------------
        gamma = self.gamma(X)                         # (K, N)
        Xc = X - self.attractor[:, None]              # x - x*
        A_vars = [cp.Variable((M, M)) for _ in range(self.K)]
        pred = sum(cp.multiply(np.tile(gamma[k], (M, 1)), A_vars[k] @ Xc)
                   for k in range(self.K))
        objective = cp.Minimize(cp.sum(cp.norm(pred - Xdot, 2, axis=0)))
        constraints = []
        for A in A_vars:
            Q = A.T @ self.P + self.P @ A
            constraints.append((Q + Q.T) / 2 << -eps * np.eye(M))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)
        if verbose:
            print(f'SDP status: {prob.status}, cost = {prob.value:.3f}')

        self.A = np.stack([A.value for A in A_vars])
        self.b = np.stack([-self.A[k] @ self.attractor for k in range(self.K)])
        return self

    # --------------------------------------------------------- evaluation --
    def gamma(self, X):
        """Mixing weights gamma_k(x) = pi_k N(x|mu_k,S_k) / sum_j pi_j N(x|mu_j,S_j).
        Computed in log-space so it never becomes 0/0 far from the data."""
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[0] != self.dim:
            X = X.T
        logp = np.stack([np.log(self.priors[k]) + gauss_logpdf(X.T, self.mu[k], self.sigma[k])
                         for k in range(self.K)])
        logp = logp.reshape(self.K, -1)
        return np.exp(logp - logsumexp(logp, axis=0, keepdims=True))

    def evaluate(self, X):
        """X: (M,) or (M, N) -> velocities with the same shape."""
        X = np.asarray(X, float)
        single = X.ndim == 1
        Xm = X.reshape(self.dim, -1)
        g = self.gamma(Xm)
        Xd = np.zeros_like(Xm)
        for k in range(self.K):
            Xd += g[k] * (self.A[k] @ Xm + self.b[k][:, None])
        return Xd[:, 0] if single else Xd

    def __call__(self, x, x0=None):
        """Signature compatible with the obstacle-avoidance package:
        dynamicalSystem(x, x0) -> xdot. x0 is ignored (the attractor is
        part of the learned model)."""
        return self.evaluate(x)

    # -------------------------------------------------------- diagnostics --
    def stability_report(self):
        """max eigenvalue of (A_k^T P + P A_k) for every k (all must be < 0)."""
        P = getattr(self, 'P', None)
        P = np.eye(self.dim) if P is None else P
        return np.array([np.linalg.eigvalsh(A.T @ P + P @ A).max() for A in self.A])

    def lyapunov(self, X):
        """V(x) = (x-x*)^T P (x-x*) and dV/dt = 2 (x-x*)^T P f(x).  X: (M, N)."""
        P = getattr(self, 'P', None)
        P = np.eye(self.dim) if P is None else P
        Xc = X - self.attractor[:, None]
        V = np.sum(Xc * (P @ Xc), axis=0)
        dV = 2 * np.sum(Xc * (P @ self.evaluate(X)), axis=0)
        return V, dV

    def simulate(self, x0, dt=0.01, n_steps=5000, tol=1e-3, modulation=None):
        """Integrates xdot = f(x) (optionally modulated: xdot = modulation(x, f(x)))
        with RK4 from x0. Returns trajectory (M, T)."""
        def rhs(x):
            v = self.evaluate(x)
            return v if modulation is None else modulation(x, v)
        x = np.asarray(x0, float).copy()
        traj = [x.copy()]
        for _ in range(n_steps):
            k1 = rhs(x); k2 = rhs(x + 0.5 * dt * k1)
            k3 = rhs(x + 0.5 * dt * k2); k4 = rhs(x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(x.copy())
            if np.linalg.norm(x - self.attractor) < tol:
                break
        return np.array(traj).T

    # ----------------------------------------------------------------- IO --
    def save(self, path):
        d = dict(priors=self.priors.tolist(), mu=self.mu.tolist(),
                 sigma=self.sigma.tolist(), A=self.A.tolist(), b=self.b.tolist(),
                 attractor=self.attractor.tolist(),
                 P=None if self.P is None else self.P.tolist())
        with open(path, 'w') as f:
            json.dump(d, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**d)
