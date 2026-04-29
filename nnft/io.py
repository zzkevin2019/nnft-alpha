"""I/O utilities: filename formatting and output-path resolution."""

import numpy as np
from pathlib import Path


def format_M(M):
    """
    Format M for filenames using scientific notation if appropriate.

    Examples:
        1000 -> "1000"
        10000 -> "1e4"
        100000000 -> "1e8"
        5000000 -> "5e6"
    """
    M = int(M)
    if M < 10000:
        return str(M)

    # Find the exponent
    exp = int(np.floor(np.log10(M)))
    mantissa = M / (10 ** exp)

    # Check if it's a clean power of 10 or simple multiple
    if abs(mantissa - round(mantissa)) < 1e-9:
        mantissa = int(round(mantissa))
        return f"{mantissa}e{exp}"
    else:
        # Not a clean number, just use the integer
        return str(M)


def _resolve_output_path(path, output_dir=None):
    """Prepend output_dir to path (unless path is absolute or output_dir is
    empty/'.'), and ensure the parent directory exists."""
    path = Path(path)
    if output_dir not in (None, '', '.') and not path.is_absolute():
        path = Path(output_dir) / path
    if path.parent != Path('.'):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _make_output_filename(d, N, Lambda, alpha, rescale, M):
    """Generate default output filename for 4-point results."""
    L_str = f"{Lambda:g}"
    a_str = f"{alpha:g}"
    r_str = f"{rescale:g}"
    M_str = format_M(M)
    return f"G4_d{d}_N{N}_L{L_str}_a{a_str}_r{r_str}_M{M_str}.npz"
