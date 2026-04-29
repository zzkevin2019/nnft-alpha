"""Generation of random 4-point coordinate configurations."""

import numpy as np

from .io import _resolve_output_path


def generate_four_point_configs(d, n_configs=50, seed=123, save_path=None, output_dir=None):
    """
    Generate random 4-point configurations with normalized scale.

    Points are drawn from a unit variance Gaussian distribution, then each
    configuration is rescaled so that its average pairwise distance equals 1.
    """
    rng = np.random.default_rng(seed)

    configs = rng.standard_normal((n_configs, 4, d)).astype(np.float32)

    for i in range(n_configs):
        dists = []
        for a in range(4):
            for b in range(a + 1, 4):
                dists.append(np.linalg.norm(configs[i, a] - configs[i, b]))
        avg_dist = np.mean(dists)

        if avg_dist > 1e-10:
            configs[i] /= avg_dist

    avg_dists = []
    for i in range(n_configs):
        dists = []
        for a in range(4):
            for b in range(a + 1, 4):
                dists.append(np.linalg.norm(configs[i, a] - configs[i, b]))
        avg_dists.append(np.mean(dists))

    print(f"Generated {n_configs} 4-point configurations in d={d}")
    print(f"  Average pairwise distance per config: "
          f"{np.mean(avg_dists):.6f} +/- {np.std(avg_dists):.6f}")

    if save_path is not None:
        save_path = _resolve_output_path(save_path, output_dir)
        np.save(save_path, configs)
        print(f"  Saved to {save_path}")

    return configs
