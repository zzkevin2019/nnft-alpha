"""Neural Network Field Theory (NNFT).

Computes correlation functions from ensembles of one-hidden-layer Cos-Net
neural networks with random parameters drawn from distributions that
reproduce free scalar field theory propagators.

All dimensionful quantities are expressed in units of the scalar mass m,
i.e. m = 1 is hardcoded throughout. Distances are in units of 1/m and
momenta (including the UV cutoff Lambda) are in units of m.

Submodules:
    theory      - Omega integrals, propagators, Wick-theorem 4-point function
    sampling    - Inverse-CDF builder and NetworkGenerator
    correlators - 2-point / 4-point estimators and parallel scan driver
    configs     - Random 4-point coordinate configuration generator
    io          - Filename formatting and output-path resolution
"""

from .io import (
    format_M,
)
from .theory import (
    compute_omega,
    propagator_theory,
    propagator_regularized,
    propagator_hybrid,
    four_point_theory,
)
from .sampling import (
    build_inverse_cdf,
    NetworkGenerator,
)
from .configs import (
    generate_four_point_configs,
)
from .correlators import (
    compute_two_point,
    compute_four_point,
    compute_four_point_parameter_scan,
)
