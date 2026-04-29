"""Field-theory predictions: Omega_alpha integrals, free-scalar propagators
(infinite-Lambda, regularized, and a hybrid for the Gaussian regulator), 
and the free-theory 4-point function via Wick's theorem."""

import numpy as np
from scipy.special import gamma, kv, jv
from scipy.integrate import quad


def compute_omega(d, alpha, Lambda, regulator='gaussian'):
    """
    Compute Omega_alpha = integral d^d w / (2*pi)^d * f_Lambda(|w|)
                          / (|w|^2 + 1)^(alpha+1)

    For alpha = 0, this equals G_2(0), the propagator at zero separation.

    regulator: 'gaussian' uses exp(-|w|^2/(2*Lambda^2)),
               'cutoff' uses theta(Lambda - |w|).
    """
    if regulator == 'gaussian':
        def radial_integrand(r):
            if r <= 0:
                return 0.0
            return (r**(d - 1)
                    * np.exp(-r**2 / (2 * Lambda**2))
                    / (r**2 + 1)**(alpha + 1))
        radial_integral, _ = quad(radial_integrand, 0, np.inf)
    elif regulator == 'cutoff':
        def radial_integrand(r):
            if r <= 0:
                return 0.0
            return (r**(d - 1)
                    / (r**2 + 1)**(alpha + 1))
        radial_integral, _ = quad(radial_integrand, 0, Lambda)
    else:
        raise ValueError(f"Unknown regulator: {regulator!r}. Use 'gaussian' or 'cutoff'.")

    angular_factor = 1.0 / (2**(d - 1) * np.pi**(d / 2) * gamma(d / 2))

    return angular_factor * radial_integral


def propagator_theory(r, d):
    """
    Exact free scalar propagator in d dimensions (Lambda -> infinity limit):

    G(r) = (1 / (2*pi)^{d/2}) * (1/r)^{d/2 - 1} * K_{d/2 - 1}(r)
    """
    if np.isscalar(r):
        if r < 1e-10:
            return np.inf
        nu = d / 2 - 1
        prefactor = 1.0 / (2 * np.pi)**(d / 2)
        return prefactor * (1 / r)**nu * kv(nu, r)
    else:
        result = np.zeros_like(r, dtype=float)
        mask = r >= 1e-10
        nu = d / 2 - 1
        prefactor = 1.0 / (2 * np.pi)**(d / 2)
        result[mask] = prefactor * (1 / r[mask])**nu * kv(nu, r[mask])
        result[~mask] = np.inf
        return result


def propagator_regularized(r, d, Lambda, regulator='gaussian'):
    """
    Regularized free scalar propagator with UV cutoff Lambda.
    Uses adaptive quadrature (scipy.quad) for high accuracy.

    regulator: 'gaussian' uses exp(-k^2/(2*Lambda^2)),
               'cutoff' uses theta(Lambda - k).
    """
    if np.isscalar(r):
        r = np.array([r])
        scalar_input = True
    else:
        r = np.asarray(r)
        scalar_input = False

    results = np.zeros_like(r, dtype=float)
    nu = d / 2 - 1
    prefactor_base = (2 * np.pi)**(-d / 2)

    if regulator == 'gaussian':
        k_max = 10 * Lambda
        def make_integrand(r_val):
            def integrand(k):
                if k <= 0:
                    return 0.0
                return (k**(d / 2)
                        * jv(nu, k * r_val)
                        * np.exp(-k**2 / (2 * Lambda**2))
                        / (k**2 + 1))
            return integrand
    elif regulator == 'cutoff':
        k_max = Lambda
        def make_integrand(r_val):
            def integrand(k):
                if k <= 0:
                    return 0.0
                return (k**(d / 2)
                        * jv(nu, k * r_val)
                        / (k**2 + 1))
            return integrand
    else:
        raise ValueError(f"Unknown regulator: {regulator!r}. Use 'gaussian' or 'cutoff'.")

    for i, r_val in enumerate(r):
        if r_val < 1e-10:
            results[i] = compute_omega(d, 0, Lambda, regulator=regulator)
        else:
            integrand = make_integrand(r_val)
            prefactor = prefactor_base * r_val**(1 - d / 2)
            integral, _ = quad(integrand, 0, k_max, limit=200)
            results[i] = prefactor * integral

    if scalar_input:
        return results[0]
    return results


def propagator_hybrid(r, d, Lambda, r_cross_factor=5.0):
    """
    Hybrid propagator for the Gaussian regulator.

    - Small r (r < r_cross_factor/Lambda): numerical integration via
      propagator_regularized.
    - Large r (r >= r_cross_factor/Lambda): analytic infinite-Lambda
      propagator corrected by exp(1/(2*Lambda^2)). The residual is
      exponentially small in (r*Lambda)^2.

    Motivation: The integrand is highly oscillatory on a smooth envelope
    extending out to k ~ several*Lambda, and at r*Lambda >> 1 scipy.quad's 
    adaptive routine can declare false convergence. The hybrid switches to 
    the analytic form precisely where direct quadrature becomes unreliable.

    Gaussian-only: the sharp-cutoff regulator has a bounded integration
    domain and the basic adaptive quadrature is reliable across the
    relevant range. The analogous large-r correction is also not a
    constant prefactor there — the difference oscillates with period
    ~2*pi/Lambda and decays only algebraically. Use
    propagator_regularized for cutoff regulator.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    result = np.zeros_like(r)
    r_cross = r_cross_factor / Lambda

    small = r < r_cross
    large = ~small

    if np.any(small):
        result[small] = propagator_regularized(r[small], d, Lambda,
                                               regulator='gaussian')
    if np.any(large):
        correction = np.exp(1 / (2 * Lambda**2))
        result[large] = propagator_theory(r[large], d) * correction

    return result if len(result) > 1 else result[0]


def four_point_theory(x1, x2, x3, x4, d, Lambda=None, hybrid=False,
                      regulator='gaussian'):
    """
    Free theory 4-point function via Wick's theorem.

    Lambda=None: infinite-Lambda propagator.
    Lambda=value, hybrid=False: full numerical regularized propagator.
    Lambda=value, hybrid=True: hybrid scheme (regularized at small r,
        corrected infinite-Lambda at large r). Only honored for
        regulator='gaussian'; for regulator='cutoff' this falls back to
        the full numerical path (the hybrid scheme is not numerically
        appropriate there — see propagator_hybrid).
    regulator: 'gaussian' or 'cutoff' (only used when Lambda is not None).
    """
    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    if Lambda is not None:
        if regulator == 'gaussian' and hybrid:
            G = lambda r: propagator_hybrid(r, d, Lambda)
        else:
            G = lambda r: propagator_regularized(r, d, Lambda,
                                                 regulator=regulator)
    else:
        G = lambda r: propagator_theory(r, d)

    G12 = G(dist(x1, x2))
    G34 = G(dist(x3, x4))
    G13 = G(dist(x1, x3))
    G24 = G(dist(x2, x4))
    G14 = G(dist(x1, x4))
    G23 = G(dist(x2, x3))

    return G12 * G34 + G13 * G24 + G14 * G23
