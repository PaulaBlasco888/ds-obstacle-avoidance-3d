"""
Numerical checks of the 3D modulation (ds_learning/obstacle3d.py).
Run:  python test_modulation.py
"""
import numpy as np
from ds_learning.obstacle3d import Ellipsoid, rotation_matrix

rng = np.random.default_rng(0)
obs = Ellipsoid(center=[0.2, -0.1, 0.3], axes=[0.3, 0.2, 0.25],
                rotation=rotation_matrix(0.3, -0.2, 0.7), sf=1.0)


def random_dirs(n):
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def surface_point(u):
    """Point on the surface (Gamma = 1) in direction u from the centre."""
    a = obs.sf * obs.a
    ul = obs.R.T @ u
    s = 1 / np.sqrt(np.sum((ul / a) ** 2))
    return obs.c + s * u


# 1) the basis [n, e1, e2] is orthonormal
errs = []
for u in random_dirs(500):
    x = obs.c + 2 * u
    _, Eo = obs.basis(x)
    errs.append(np.abs(Eo.T @ Eo - np.eye(3)).max())
print(f'1) max |E_orth^T E_orth - I|           = {max(errs):.1e}   (orthonormal basis)')

# 2) Gamma = 1 on the surface and n = grad Gamma (finite differences)
g_err, n_err = [], []
for u in random_dirs(500):
    xs = surface_point(u)
    g_err.append(abs(obs.gamma(xs) - 1))
    h = 1e-6
    grad = np.array([(obs.gamma(xs + h * e) - obs.gamma(xs - h * e)) / (2 * h) for e in np.eye(3)])
    n_err.append(np.linalg.norm(grad / np.linalg.norm(grad) - obs.normal(xs)))
print(f'2) max |Gamma(surface) - 1|            = {max(g_err):.1e}')
print(f'   max |n - gradGamma/|gradGamma||     = {max(n_err):.1e}')

# 3) impenetrability: on the surface, the modulated velocity has no component
#    along the normal (for any f pointing towards the obstacle)
pen = []
for u in random_dirs(500):
    xs = surface_point(u) + 1e-9 * u
    f = -obs.normal(xs) + 0.5 * rng.normal(size=3)       # velocity "into" the obstacle
    if np.dot(f, obs.normal(xs)) >= 0:
        continue
    v = obs.modulate(xs, f)
    pen.append(abs(np.dot(v, obs.normal(xs))) / np.linalg.norm(f))
print(f'3) max |n . M f| / |f| on the surface   = {max(pen):.1e}   (no penetration)')

# 4) far away M -> I
far = [np.abs(obs.modulation_matrix(obs.c + 1e3 * u) - np.eye(3)).max() for u in random_dirs(100)]
print(f'4) max |M - I| at distance 1000         = {max(far):.1e}   (DS unchanged far away)')

assert max(errs) < 1e-10 and max(g_err) < 1e-10 and max(n_err) < 1e-5
assert max(pen) < 1e-6 and max(far) < 1e-2
print('All checks passed.')
