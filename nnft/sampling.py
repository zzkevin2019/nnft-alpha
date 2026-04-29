"""Network parameter sampling: inverse-CDF builder and NetworkGenerator.

Handles the non-trivial part of drawing (w1, b1, w2) for each neuron so that
the resulting Cos-Net reproduces the target free-scalar propagator for a
chosen (alpha, Lambda, regulator).
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .theory import compute_omega


def build_inverse_cdf(d, alpha, Lambda, n_grid=2000, regulator='gaussian'):
    """
    Build inverse CDF for sampling |w1| from the distribution:
    p(|w1|) proportional to r^{d-1} f_Lambda / (r^2 + 1)^{alpha+1}

    where f_Lambda is exp(-r^2/(2*Lambda^2)) for Gaussian regulator
    or theta(Lambda - r) for cutoff regulator.

    Uses scipy.integrate.quad for accurate forward CDF computation at each
    grid point, with a hybrid grid (fine near origin, coarser at large r).
    """
    if regulator == 'gaussian':
        def radial_density(r):
            if r <= 0:
                return 0.0
            return (r**(d - 1)
                    * np.exp(-r**2 / (2 * Lambda**2))
                    / (r**2 + 1)**(alpha + 1))
        total, _ = quad(radial_density, 0, np.inf)
    elif regulator == 'cutoff':
        def radial_density(r):
            if r <= 0:
                return 0.0
            return (r**(d - 1)
                    / (r**2 + 1)**(alpha + 1))
        total, _ = quad(radial_density, 0, Lambda)
    else:
        raise ValueError(f"Unknown regulator: {regulator!r}. Use 'gaussian' or 'cutoff'.")

    # Find r_max such that CDF(r_max) is close to 1
    if regulator == 'cutoff':
        r_max = Lambda
    else:
        r_max = Lambda
        while True:
            cdf_val, _ = quad(radial_density, 0, r_max)
            cdf_val /= total
            if cdf_val > 1 - 1e-12 or r_max > 50 * Lambda:
                break
            r_max *= 1.5

    # Use a grid with finer spacing near origin to capture d=1 behavior
    # where the density doesn't vanish at r=0
    n_fine = n_grid // 2
    n_coarse = n_grid - n_fine

    r_fine_max = min(10, r_max / 10)
    r_fine = np.linspace(0, r_fine_max, n_fine)

    r_coarse = np.linspace(r_fine_max, r_max, n_coarse + 1)[1:] # exclude duplicate point

    r_grid = np.concatenate([r_fine, r_coarse])

    cdf_grid = np.zeros(len(r_grid))
    for i, r_val in enumerate(r_grid):
        if r_val <= 0:
            cdf_grid[i] = 0.0
        else:
            cdf_grid[i], _ = quad(radial_density, 0, r_val)
    cdf_grid /= total

    # Ensure strict monotonicity (numerical noise can cause tiny violations)
    for i in range(1, len(cdf_grid)):
        if cdf_grid[i] <= cdf_grid[i-1]:
            cdf_grid[i] = cdf_grid[i-1] + 1e-15

    return interp1d(cdf_grid, r_grid, kind='linear',
                    bounds_error=False, fill_value=(0, r_grid[-1]))


class NetworkGenerator:
    """
    Generator for NNFT network parameters.

    Special case: When alpha = -1 and regulator = 'gaussian',
    w1 is simply Gaussian with variance Lambda^2.
    When alpha = -1 and regulator = 'cutoff',
    w1 is uniform in a ball of radius Lambda.
    """

    def __init__(self, d, alpha=0, N=100, Lambda=100.0, regulator='gaussian'):
        self.d = d
        self.alpha = alpha
        self.N = N
        self.Lambda = Lambda
        self.regulator = regulator

        self.omega = compute_omega(d, self.alpha, self.Lambda, regulator=regulator)

        if alpha == -1:
            self.inverse_cdf = None
        else:
            self.inverse_cdf = build_inverse_cdf(d, self.alpha, self.Lambda,
                                                 regulator=regulator)

    def generate_batch(self, batch_size, rng):
        N, d = self.N, self.d

        if self.alpha == -1 and self.regulator == 'gaussian':
            W1 = (self.Lambda * rng.standard_normal((batch_size, N, d))).astype(np.float32)
            radii = np.linalg.norm(W1, axis=2)
        elif self.alpha == -1 and self.regulator == 'cutoff':
            directions = rng.standard_normal((batch_size, N, d)).astype(np.float32)
            directions /= np.linalg.norm(directions, axis=2, keepdims=True)
            u = rng.random((batch_size, N)).astype(np.float32)
            radii = (self.Lambda * u**(1.0 / d)).astype(np.float32)
            W1 = radii[:, :, np.newaxis] * directions
        else:
            u = rng.random((batch_size, N))
            radii = self.inverse_cdf(u).astype(np.float32)

            directions = rng.standard_normal((batch_size, N, d)).astype(np.float32)
            directions /= np.linalg.norm(directions, axis=2, keepdims=True)

            W1 = radii[:, :, np.newaxis] * directions

        b1 = rng.uniform(-np.pi, np.pi, (batch_size, N)).astype(np.float32)
        w2 = (np.sqrt(2 * self.omega) * (radii**2 + 1) ** (self.alpha / 2)).astype(np.float32)

        return W1, b1, w2
