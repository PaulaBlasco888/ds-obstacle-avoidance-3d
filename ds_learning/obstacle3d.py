"""
3D obstacle modelling and modulation of a dynamical system
(Khansari-Zadeh & Billard 2012; Huber, Billard & Slotine 2019).

    xdot_mod = M(x) f(x),        M(x) = E(x) D(x) E(x)^-1   in R^{3x3}

Obstacle (ellipsoid, or "superellipsoid" with p > 1), in its local frame
x_l = R^T (x - c):

    Gamma(x) = sum_i ( x_l,i / (sf * a_i) )^(2 p_i)      Gamma = 1 on the surface,
                                                         > 1 outside, < 1 inside
    n(x)     = grad Gamma / |grad Gamma|                 outward surface normal
    e1, e2   : orthonormal tangent vectors (n, e1, e2 orthonormal basis of R^3)
    r(x)     = (x - x_ref) / |x - x_ref|                 reference direction
    E(x)     = [ r  e1  e2 ]
    D(x)     = diag( 1 - 1/Gamma^(1/rho),  1 + 1/Gamma^(1/rho),  1 + 1/Gamma^(1/rho) )

On the surface (Gamma = 1) lambda_r = 0: the velocity component that would
enter the obstacle is removed and only the tangential motion remains -> the
obstacle is never penetrated (impenetrability). Far away (Gamma -> inf)
D -> I, M -> I and the original DS is recovered.

With the reference point inside a star-shaped obstacle, E is invertible
everywhere outside the obstacle and convergence to the attractor is kept
(Huber et al. 2019). For a sphere centered at x_ref, r = n.
"""
import numpy as np


def rotation_matrix(roll=0.0, pitch=0.0, yaw=0.0):
    """R = Rz(yaw) Ry(pitch) Rx(roll), angles in radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def orthonormal_basis(n):
    """Completes the unit vector n to a right-handed orthonormal basis
    [n, e1, e2] of R^3 (e1, e2 span the tangent plane)."""
    n = n / np.linalg.norm(n)
    # pick the canonical axis least aligned with n to avoid degeneracy
    helper = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


class Ellipsoid:
    def __init__(self, center, axes, rotation=None, p=1, sf=1.0, rho=1.0, reference_point=None):
        """
        center   : (3,) centre c of the obstacle
        axes     : (3,) semi-axes a
        rotation : (3,3) orientation R (default identity)
        p        : curvature (1 = ellipsoid; >1 more box-like)
        sf       : safety factor (the modulation sees axes sf*a)
        rho      : reactivity (>1 the obstacle is "felt" from further away)
        reference_point : x_ref inside the obstacle (default: centre)
        """
        self.c = np.asarray(center, float)
        self.a = np.asarray(axes, float)
        self.R = np.eye(3) if rotation is None else np.asarray(rotation, float)
        self.p = np.broadcast_to(np.asarray(p, float), (3,)).copy()
        self.sf = float(sf)
        self.rho = float(rho)
        self.x_ref = self.c.copy() if reference_point is None else np.asarray(reference_point, float)

    # ---------------- geometry ------------------------------------------------
    def to_local(self, x):
        return self.R.T @ (np.asarray(x, float) - self.c)

    def gamma(self, x, with_margin=True):
        a = self.sf * self.a if with_margin else self.a
        xl = self.to_local(x)
        return float(np.sum((xl / a) ** (2 * self.p)))

    def normal(self, x):
        """Outward unit normal n(x) = grad Gamma / |grad Gamma| (world frame)."""
        a = self.sf * self.a
        xl = self.to_local(x)
        grad_local = 2 * self.p / a * (xl / a) ** (2 * self.p - 1)
        g = self.R @ grad_local
        nrm = np.linalg.norm(g)
        return g / nrm if nrm > 0 else np.array([1.0, 0.0, 0.0])

    def reference_direction(self, x):
        d = np.asarray(x, float) - self.x_ref
        nrm = np.linalg.norm(d)
        return d / nrm if nrm > 0 else self.normal(x)

    # ---------------- modulation ---------------------------------------------
    def basis(self, x):
        """E(x) = [r, e1, e2] and the orthonormal basis E_orth = [n, e1, e2]."""
        n, e1, e2 = orthonormal_basis(self.normal(x))
        r = self.reference_direction(x)
        E = np.column_stack([r, e1, e2])
        E_orth = np.column_stack([n, e1, e2])
        return E, E_orth

    def eigenvalues(self, x, f=None, tail_effect=False):
        G = max(self.gamma(x), 1.0)
        lam_r = 1.0 - 1.0 / G ** (1.0 / self.rho)
        lam_e = 1.0 + 1.0 / G ** (1.0 / self.rho)
        # Without tail effect: once the robot moves away from the obstacle
        # (f . n > 0) the normal component is not reduced any more.
        if f is not None and not tail_effect and np.dot(self.normal(x), f) > 0:
            lam_r = 1.0
        return np.array([lam_r, lam_e, lam_e])

    def modulation_matrix(self, x, f=None, tail_effect=False):
        E, _ = self.basis(x)
        D = np.diag(self.eigenvalues(x, f, tail_effect))
        return E @ D @ np.linalg.inv(E)

    def modulate(self, x, f, tail_effect=False):
        """xdot_mod = M(x) f(x). Inside the obstacle (numerical safety only)
        a repulsive velocity along r is returned."""
        f = np.asarray(f, float)
        if self.gamma(x) < 1.0:
            return self.reference_direction(x) * max(np.linalg.norm(f), 0.05)
        return self.modulation_matrix(x, f, tail_effect) @ f

    # ---------------- drawing --------------------------------------------------
    def surface(self, n=40, with_margin=False):
        """Mesh (X, Y, Z) of the obstacle surface for plot_surface."""
        a = self.sf * self.a if with_margin else self.a
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n))
        # superellipsoid parametrisation (p = 1 -> ellipsoid)
        e = 1.0 / self.p
        def spow(w, k):
            return np.sign(w) * np.abs(w) ** k
        xl = np.stack([a[0] * spow(np.cos(u), e[0]) * spow(np.sin(v), e[0]),
                       a[1] * spow(np.sin(u), e[1]) * spow(np.sin(v), e[1]),
                       a[2] * spow(np.cos(v), e[2])])
        xw = np.einsum('ij,jkl->ikl', self.R, xl) + self.c[:, None, None]
        return xw[0], xw[1], xw[2]
