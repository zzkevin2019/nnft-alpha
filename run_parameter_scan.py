"""
Parameter scan script for NNFT 4-point function calculations.

Scans over alpha and Lambda for fixed d and N.
Uses compute_four_point_parameter_scan for efficient parallelization.

Usage:
    python run_parameter_scan.py [options]
    
Examples:
    python run_parameter_scan.py --d 3 --N 100 --M 1e6
    python run_parameter_scan.py --d 2 --N 50 --alphas -1 0 1 --Lambdas 10 100 --M 1e8
"""

import argparse
import time
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Import from nnft.py (assumes it's in the same directory or on PYTHONPATH)
from nnft import (
    generate_four_point_configs,
    compute_four_point_parameter_scan,
    format_M,
)


def create_tasks(d, N, alphas, Lambdas, rescale_values, n_configs, config_seed,
                 regulator='gaussian'):
    """
    Create task list for parameter scan.

    All tasks share the same n_configs random 4-point configurations.
    """
    configs = generate_four_point_configs(
        d=d, n_configs=n_configs, seed=config_seed, save_path=None
    )

    tasks = []

    for alpha in alphas:
        for Lambda in Lambdas:
            tasks.append({
                'd': d,
                'N': N,
                'alpha': alpha,
                'Lambda': Lambda,
                'regulator': regulator,
                'configs': configs,
                'rescale_values': rescale_values,
            })

    return tasks


def run_scan(d, N, M, alphas, Lambdas, rescale_values, n_configs,
             n_workers, output_dir, seed, config_seed, batch_size, config_chunk,
             regulator='gaussian'):
    """
    Run the parameter scan and return timing information.
    """
    print("=" * 70)
    print("NNFT Parameter Scan")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters:")
    print(f"  d = {d}")
    print(f"  N = {N}")
    print(f"  M = {format_M(M)}")
    print(f"  alphas = {alphas}")
    print(f"  Lambdas = {Lambdas}")
    print(f"  rescale_values = {rescale_values}")
    print(f"  n_configs = {n_configs}")
    print(f"  n_workers = {n_workers}")
    print(f"  batch_size = {batch_size}")
    print(f"  config_chunk = {config_chunk}")
    print(f"  seed = {seed}")
    print(f"  config_seed = {config_seed}")
    print(f"  regulator = {regulator}")
    print(f"  output_dir = {output_dir}")
    print()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create tasks
    print("Creating tasks...")
    t_start_tasks = time.perf_counter()
    tasks = create_tasks(d, N, alphas, Lambdas, rescale_values, n_configs, config_seed,
                         regulator=regulator)
    t_tasks = time.perf_counter() - t_start_tasks
    print(f"  Created {len(tasks)} tasks in {t_tasks:.2f}s")
    print()
    
    # Snapshot existing files before the scan
    output_path = Path(output_dir)
    existing_files = set(output_path.glob("G4_*.npz")) if output_path.exists() else set()

    # Run the scan
    print("Running parameter scan...")
    t_start_scan = time.perf_counter()

    results, worker_timings = compute_four_point_parameter_scan(
        tasks=tasks,
        M=M,
        seed=seed,
        batch_size=batch_size,
        config_chunk=config_chunk,
        n_workers=n_workers,
        output_dir=output_dir,
        save_results=True,
        verbose=True,
    )

    t_scan = time.perf_counter() - t_start_scan
    t_total = t_tasks + t_scan

    print()
    print("=" * 70)
    print("Wall-clock Summary")
    print("=" * 70)
    print(f"  Task creation:    {t_tasks:8.2f}s")
    print(f"  Computation:      {t_scan:8.2f}s")
    print(f"  Total:            {t_total:8.2f}s")
    print()

    # Per-task timing estimate
    n_tasks = len(tasks)
    n_rescales = len(rescale_values)
    total_computations = n_tasks * n_rescales
    print(f"  Tasks:            {n_tasks}")
    print(f"  Rescale values:   {n_rescales}")
    print(f"  Total outputs:    {total_computations}")
    print(f"  Avg per output:   {t_scan/total_computations:.2f}s")
    print()

    # Save timing info to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timing_file = Path(output_dir) / f"timing_d{d}_N{N}_M{format_M(M)}_{timestamp}.txt"
    with open(timing_file, 'w') as f:
        f.write(f"NNFT Parameter Scan Timing\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  d = {d}\n")
        f.write(f"  N = {N}\n")
        f.write(f"  M = {format_M(M)}\n")
        f.write(f"  alphas = {alphas}\n")
        f.write(f"  Lambdas = {Lambdas}\n")
        f.write(f"  rescale_values = {rescale_values}\n")
        f.write(f"  n_configs = {n_configs}\n")
        f.write(f"  n_workers = {n_workers}\n")
        f.write(f"  batch_size = {batch_size}\n")
        f.write(f"  config_chunk = {config_chunk}\n")
        f.write(f"  seed = {seed}\n")
        f.write(f"  config_seed = {config_seed}\n\n")
        f.write(f"Wall-clock timing:\n")
        f.write(f"  Task creation:    {t_tasks:8.2f}s\n")
        f.write(f"  Computation:      {t_scan:8.2f}s\n")
        f.write(f"  Total:            {t_total:8.2f}s\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Tasks:            {n_tasks}\n")
        f.write(f"  Rescale values:   {n_rescales}\n")
        f.write(f"  Total outputs:    {total_computations}\n")
        f.write(f"  Avg per output:   {t_scan/total_computations:.2f}s\n\n")
        f.write(f"Per-worker timing breakdown (seconds):\n")
        f.write(f"  {'Task':>40s}  {'Init':>6s}  {'Gen':>7s}  {'G4':>7s}  "
                f"{'Theory':>7s}  {'Total':>7s}\n")
        f.write(f"  {'-'*40}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}\n")
        for task_key, t in worker_timings.items():
            dk, Nk, alphak, Lambdak = task_key
            label = f"d={dk} N={Nk} alpha={alphak} Lambda={Lambdak}"
            f.write(f"  {label:>40s}  {t['init']:6.1f}  "
                    f"{t['ensemble_gen']:7.1f}  {t['g4_compute']:7.1f}  "
                    f"{t['theory']:7.1f}  {t['total']:7.1f}\n")

    print(f"Timing saved to: {timing_file}")
    
    # List only new output files from this run
    print()
    all_files = set(output_path.glob("G4_*.npz"))
    new_files = sorted(all_files - existing_files)
    print(f"New output files ({len(new_files)}):")
    for f in new_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")
    
    return results, {
        't_tasks': t_tasks,
        't_scan': t_scan,
        't_total': t_total,
        'n_tasks': n_tasks,
        'n_rescales': n_rescales,
        'worker_timings': worker_timings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run NNFT parameter scan over alpha and Lambda",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required parameters
    parser.add_argument('--d', type=int, default=2,
                        help='Spatial dimension')
    parser.add_argument('--N', type=int, default=100,
                        help='Number of hidden units')
    parser.add_argument('--M', type=float, default=1e8,
                        help='Ensemble size (number of networks)')
    
    # Scan parameters
    parser.add_argument('--alphas', type=float, nargs='+', default=[-1, 0, 1],
                        help='Alpha values to scan')
    parser.add_argument('--Lambdas', type=float, nargs='+', default=[100],
                        help='Lambda values to scan')
    parser.add_argument('--rescales', type=float, nargs='+', 
                        default=[0.3, 1.0, 3.0],
                        help='Rescale values')
    
    # Configuration parameters
    parser.add_argument('--n_configs', type=int, default=50,
                        help='Number of 4-point configurations per task')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: number of tasks)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for network generation')
    
    # Seeds
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for ensemble generation')
    parser.add_argument('--config_seed', type=int, default=123,
                        help='Random seed for configuration generation')
    
    # Regulator
    parser.add_argument('--regulator', type=str, default='gaussian',
                        choices=['gaussian', 'cutoff'],
                        help='UV regulator type')

    # Output
    parser.add_argument('--output_dir', type=str, default='data/scan',
                        help='Output directory for results')
    
    # Chunking
    parser.add_argument('--config_chunk', type=int, default=10,
                        help='Number of configs to process together per batch')
    
    args = parser.parse_args()
    
    # Run the scan
    results, timing = run_scan(
        d=args.d,
        N=args.N,
        M=args.M,
        alphas=args.alphas,
        Lambdas=args.Lambdas,
        rescale_values=args.rescales,
        n_configs=args.n_configs,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        seed=args.seed,
        config_seed=args.config_seed,
        batch_size=args.batch_size,
        config_chunk=args.config_chunk,
        regulator=args.regulator,
    )
    
    print()
    print("Done!")
    
    return results, timing


if __name__ == '__main__':
    main()
