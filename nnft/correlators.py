"""Monte Carlo estimators for 2-point and 4-point correlation functions,
including a parameter-set parallelization driver."""

import os
import time
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from .io import format_M, _resolve_output_path, _make_output_filename
from .sampling import NetworkGenerator
from .theory import four_point_theory


def compute_two_point(configs, d, M, seed=42, alpha=0, N=100, Lambda=100.0,
                      batch_size=100, verbose=True, regulator='gaussian'):
    """
    Compute the 2-point function <phi(x1) phi(x2)> (serial version).
    """
    M = int(M)
    configs = np.asarray(configs, dtype=np.float32)
    n_configs = len(configs)

    if configs.shape != (n_configs, 2, d):
        raise ValueError(f"configs must have shape (n_configs, 2, {d}), got {configs.shape}")

    gen = NetworkGenerator(d, alpha, N, Lambda, regulator=regulator)
    rng = np.random.default_rng(seed)

    sqrt_N = np.sqrt(gen.N)
    n_batches = (M + batch_size - 1) // batch_size

    sum_G2 = np.zeros(n_configs, dtype=np.float64)
    sum_G2_sq = np.zeros(n_configs, dtype=np.float64)

    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="G2", unit="batch",
                          total=n_batches, leave=True,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for b in batch_iter:
        current_batch = min(batch_size, M - b * batch_size)
        W1, b1, w2 = gen.generate_batch(current_batch, rng)

        phi = np.zeros((2, n_configs, current_batch), dtype=np.float32)
        for p in range(2):
            pre = np.einsum('cd,mnd->cmn', configs[:, p, :], W1) + b1
            phi[p] = np.einsum('cmn,mn->cm', np.cos(pre), w2) / sqrt_N

        G2_batch = phi[0] * phi[1]
        sum_G2 += np.sum(G2_batch, axis=1)
        sum_G2_sq += np.sum(G2_batch**2, axis=1)

    means = sum_G2 / M
    variance = (sum_G2_sq / M - means**2) * M / (M - 1)
    errors = np.sqrt(variance / M)

    return means, errors


def compute_four_point(configs, d, M, seed=42, alpha=0, N=100, Lambda=100.0,
                       batch_size=100, config_chunk=10, rescale=1.0, verbose=True,
                       regulator='gaussian'):
    """
    Compute the 4-point function <phi(x1) phi(x2) phi(x3) phi(x4)> (serial version).
    """
    M = int(M)

    if isinstance(configs, (str, Path)):
        configs = np.load(configs)

    configs = np.asarray(configs, dtype=np.float32)
    n_configs = len(configs)

    if configs.shape != (n_configs, 4, d):
        raise ValueError(f"configs must have shape (n_configs, 4, {d}), got {configs.shape}")

    if rescale != 1.0:
        configs = configs * rescale
        if verbose:
            print(f"Rescaled coordinates by factor {rescale}")

    gen = NetworkGenerator(d, alpha, N, Lambda, regulator=regulator)
    rng = np.random.default_rng(seed)

    sqrt_N = np.sqrt(gen.N)
    n_batches = (M + batch_size - 1) // batch_size

    sum_G4 = np.zeros(n_configs, dtype=np.float64)
    sum_G4_sq = np.zeros(n_configs, dtype=np.float64)

    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="G4", unit="batch",
                          total=n_batches, leave=True,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for b in batch_iter:
        current_batch = min(batch_size, M - b * batch_size)
        W1, b1, w2 = gen.generate_batch(current_batch, rng)

        for c_start in range(0, n_configs, config_chunk):
            c_end = min(c_start + config_chunk, n_configs)
            chunk = configs[c_start:c_end]
            chunk_size = c_end - c_start

            phi_product = np.ones((chunk_size, current_batch), dtype=np.float32)
            for p in range(4):
                pre = np.einsum('cd,mnd->cmn', chunk[:, p, :], W1) + b1
                phi_p = np.einsum('cmn,mn->cm', np.cos(pre), w2) / sqrt_N
                phi_product *= phi_p

            sum_G4[c_start:c_end] += np.sum(phi_product, axis=1)
            sum_G4_sq[c_start:c_end] += np.sum(phi_product**2, axis=1)

    means = sum_G4 / M
    variance = (sum_G4_sq / M - means**2) * M / (M - 1)
    errors = np.sqrt(variance / M)

    if verbose:
        print("Computing theory predictions...")
    theory_reg = four_point_theory(configs[:, 0], configs[:, 1],
                                    configs[:, 2], configs[:, 3],
                                    d, Lambda=Lambda, hybrid=True,
                                    regulator=regulator)
    theory_inf = four_point_theory(configs[:, 0], configs[:, 1],
                                    configs[:, 2], configs[:, 3], d, Lambda=None)

    return means, errors, theory_reg, theory_inf


def _worker_parameter_set(args):
    """
    Worker function for parameter-set parallelization.

    Each worker:
    1. Receives a task dict containing params (d, N, alpha, Lambda), configs,
       and rescale_values
    2. Generates its ensemble ONCE (batch by batch)
    3. For each batch, computes G4 contributions for ALL rescale values

    This is the most efficient approach because ensemble generation is not duplicated
    across rescale values.
    """
    (task, M, seed, batch_size, config_chunk, output_dir, save_results, worker_id) = args

    d = task['d']
    N = task['N']
    alpha = task['alpha']
    Lambda = task['Lambda']
    regulator = task.get('regulator', 'gaussian')
    configs = np.asarray(task['configs'], dtype=np.float32)
    # Use float32 for rescale_values to prevent dtype promotion when scaling configs
    rescale_values = np.asarray(task['rescale_values'], dtype=np.float32)

    n_configs = len(configs)
    n_batches = (M + batch_size - 1) // batch_size

    configs_scaled = {r: configs * r for r in rescale_values}

    t_init_start = time.perf_counter()
    gen = NetworkGenerator(d, alpha, N, Lambda, regulator=regulator)
    rng = np.random.default_rng(seed)
    sqrt_N = np.sqrt(gen.N)
    t_init = time.perf_counter() - t_init_start

    sums_G4 = {r: np.zeros(n_configs, dtype=np.float64) for r in rescale_values}
    sums_G4_sq = {r: np.zeros(n_configs, dtype=np.float64) for r in rescale_values}

    # Main loop: generate each batch once, use for all rescale values
    t_gen_total = 0.0
    t_g4_total = 0.0
    progress_interval = 10000 # print progress every this many batches
    t_loop_start = time.perf_counter()
    for b in range(n_batches):
        current_batch = min(batch_size, M - b * batch_size)

        t0 = time.perf_counter()
        W1, b1, w2 = gen.generate_batch(current_batch, rng)
        t_gen_total += time.perf_counter() - t0

        # Progress reporting
        if (b + 1) % progress_interval == 0 or b == n_batches - 1:
            frac = (b + 1) / n_batches
            elapsed = time.perf_counter() - t_loop_start
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0
            now = datetime.now().strftime('%H:%M:%S')
            eta_str = str(timedelta(seconds=int(eta)))
            print(f"  [Worker {worker_id}] d={d} N={N} alpha={alpha} "
                  f"Lambda={Lambda}: batch {b+1}/{n_batches} "
                  f"({frac*100:.1f}%) at {now}, ETA {eta_str}",
                  flush=True)

        # For each rescale value, compute G4 using this batch
        t0 = time.perf_counter()
        for rescale in rescale_values:
            cfg = configs_scaled[rescale]

            for c_start in range(0, n_configs, config_chunk):
                c_end = min(c_start + config_chunk, n_configs)
                chunk = cfg[c_start:c_end]
                chunk_size = c_end - c_start

                phi_product = np.ones((chunk_size, current_batch), dtype=np.float32)
                for p in range(4):
                    pre = np.einsum('cd,mnd->cmn', chunk[:, p, :], W1) + b1
                    phi_p = np.einsum('cmn,mn->cm', np.cos(pre), w2) / sqrt_N
                    phi_product *= phi_p

                sums_G4[rescale][c_start:c_end] += np.sum(phi_product, axis=1)
                sums_G4_sq[rescale][c_start:c_end] += np.sum(phi_product**2, axis=1)
        t_g4_total += time.perf_counter() - t0

    # Compute final results for all rescale values
    t_theory_start = time.perf_counter()
    results = {}
    for rescale in rescale_values:
        means = sums_G4[rescale] / M
        variance = (sums_G4_sq[rescale] / M - means**2) * M / (M - 1)
        errors = np.sqrt(variance / M)

        cfg = configs_scaled[rescale]
        theory_inf = four_point_theory(cfg[:, 0], cfg[:, 1],
                                       cfg[:, 2], cfg[:, 3],
                                       d, Lambda=None)
        theory_reg = four_point_theory(cfg[:, 0], cfg[:, 1],
                                       cfg[:, 2], cfg[:, 3],
                                       d, Lambda=Lambda, hybrid=True,
                                       regulator=regulator)

        results[rescale] = {
            'means': means,
            'errors': errors,
            'theory_reg': theory_reg,
            'theory_inf': theory_inf,
            'configs': cfg,
        }

        if save_results and output_dir is not None:
            output_path = _make_output_filename(d, N, Lambda, alpha, rescale, M)
            output_path = _resolve_output_path(output_path, output_dir)

            np.savez(output_path,
                     configs=cfg,
                     means=means,
                     errors=errors,
                     theory_reg=theory_reg,
                     theory_inf=theory_inf,
                     d=d,
                     M=M,
                     seed=seed,
                     alpha=alpha,
                     N=N,
                     Lambda=Lambda,
                     rescale=rescale,
                     regulator=regulator)

    t_theory = time.perf_counter() - t_theory_start

    task_key = (d, N, alpha, Lambda)
    timing = {
        'init': t_init,
        'ensemble_gen': t_gen_total,
        'g4_compute': t_g4_total,
        'theory': t_theory,
        'total': t_init + t_gen_total + t_g4_total + t_theory,
    }
    return task_key, results, timing


def compute_four_point_parameter_scan(tasks, M, seed=42, batch_size=100,
                                       config_chunk=10, n_workers=None,
                                       output_dir=None, save_results=False,
                                       verbose=True):
    """
    Compute 4-point function for multiple tasks in parallel.

    This is the most efficient parallelization strategy because each worker
    generates its own ensemble ONCE and then computes G4 for all configs and
    all rescale values specified in its task. No redundant ensemble generation.

    Args:
        tasks: List of task dicts. Each task must contain:
               - 'd': spatial dimension
               - 'N': number of hidden units
               - 'alpha': power-law exponent
               - 'Lambda': UV cutoff
               - 'configs': array of shape (n_configs, 4, d), or path to .npy file
               - 'rescale_values': list/array of rescale factors for this task
               Optional:
               - 'regulator': 'gaussian' (default) or 'cutoff'

        M: Number of networks in ensemble (can be float like 1e8)
        seed: Random seed for reproducibility (each worker uses same seed)
        batch_size: Number of networks per batch
        config_chunk: Number of configs to process together
        n_workers: Number of parallel workers (default: number of tasks)
        output_dir: Directory for output files (if save_results=True)
        save_results: Whether to save results to files
        verbose: Print progress

    Returns:
        all_results: Dict mapping (d, N, alpha, Lambda) tuple to dict of results,
                     where each result dict maps rescale -> {means, errors, ...}
        all_timings: Dict mapping (d, N, alpha, Lambda) tuple to timing dict with
                     keys: init, ensemble_gen, g4_compute, theory, total
    """
    M = int(M)
    n_tasks = len(tasks)

    required_keys = ['d', 'N', 'alpha', 'Lambda', 'configs', 'rescale_values']
    processed_tasks = []
    for i, task in enumerate(tasks):
        for key in required_keys:
            if key not in task:
                raise ValueError(f"Task {i} missing required key: {key}")

        configs = task['configs']
        if isinstance(configs, (str, Path)):
            configs = np.load(configs)
        configs = np.asarray(configs, dtype=np.float32)

        d = task['d']
        n_configs = len(configs)
        if configs.shape != (n_configs, 4, d):
            raise ValueError(f"Task {i}: configs must have shape (n_configs, 4, {d}), "
                           f"got {configs.shape}")

        processed_tasks.append({
            'd': task['d'],
            'N': task['N'],
            'alpha': task['alpha'],
            'Lambda': task['Lambda'],
            'regulator': task.get('regulator', 'gaussian'),
            'configs': configs,
            'rescale_values': np.asarray(task['rescale_values'], dtype=np.float32),
        })

    if n_workers is None:
        n_workers = os.cpu_count()
    n_workers = min(n_workers, n_tasks)

    if verbose:
        print(f"Parameter-set parallel scan: {n_tasks} tasks")
        print(f"  M={format_M(M)}, using {n_workers} parallel workers")
        print(f"  Each worker generates ensemble ONCE, then computes all rescales")
        for i, task in enumerate(processed_tasks):
            r_list = [float(r) for r in task['rescale_values']]
            print(f"  Task {i}: d={task['d']}, N={task['N']}, "
                  f"alpha={task['alpha']}, Lambda={task['Lambda']}, "
                  f"{len(task['configs'])} configs, r={r_list}")
        if save_results and output_dir:
            print(f"  Saving results to: {output_dir}")

    worker_args = [
        (processed_tasks[i], M, seed, batch_size, config_chunk,
         output_dir, save_results, i)
        for i in range(n_tasks)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_list = list(executor.map(_worker_parameter_set, worker_args))

    all_results = {}
    all_timings = {}
    for task_key, results, timing in results_list:
        all_results[task_key] = results
        all_timings[task_key] = timing

    if verbose:
        print(f"  Done.")
        print()
        print("  Per-worker timing breakdown:")
        print(f"  {'Task':>40s}  {'Init':>6s}  {'Gen':>7s}  {'G4':>7s}  "
              f"{'Theory':>7s}  {'Total':>7s}")
        print(f"  {'-'*40}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for task_key, t in all_timings.items():
            d, N, alpha, Lambda = task_key
            label = f"d={d} N={N} alpha={alpha} Lambda={Lambda}"
            print(f"  {label:>40s}  {t['init']:6.1f}  "
                  f"{t['ensemble_gen']:7.1f}  {t['g4_compute']:7.1f}  "
                  f"{t['theory']:7.1f}  {t['total']:7.1f}")

    return all_results, all_timings
