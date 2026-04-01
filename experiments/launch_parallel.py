#!/usr/bin/env python3
"""Launch experiments in parallel across multiple GPUs.

Distributes tasks or suites across available GPUs, running one
independent process per GPU. Each process loads its own model.

Examples:
    # Grid ablation on 4 suites across 4 GPUs
    python experiments/launch_parallel.py grid_ablation --model xvla \\
        --suites libero_goal libero_object libero_spatial libero_10 \\
        --gpus 0 1 2 3

    # Split tasks across 2 GPUs
    python experiments/launch_parallel.py baseline --model smolvla \\
        --suite libero_object --split-tasks --gpus 0 1 --n-episodes 3

    # Vision perturbation on 8 GPUs
    python experiments/launch_parallel.py vision_perturbation --model groot \\
        --suites libero_goal libero_object --gpus 0 1 2 3 4 5 6 7
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch experiments across GPUs")
    parser.add_argument("experiment", help="Experiment script name (e.g. grid_ablation)")
    parser.add_argument("--gpus", type=int, nargs="+", required=True, help="GPU indices")
    parser.add_argument("--suites", type=str, nargs="+", default=None,
                        help="Suites to distribute (one per GPU)")
    parser.add_argument("--split-tasks", action="store_true",
                        help="Split tasks across GPUs instead of suites")
    parser.add_argument("--n-tasks", type=int, default=10,
                        help="Total tasks to split (with --split-tasks)")
    # Pass all other args through
    args, extra_args = parser.parse_known_args()

    script = Path(__file__).parent / f"{args.experiment}.py"
    if not script.exists():
        print(f"Script not found: {script}")
        sys.exit(1)

    processes = []

    if args.suites:
        for i, suite in enumerate(args.suites):
            gpu = args.gpus[i % len(args.gpus)]
            # CUDA_VISIBLE_DEVICES remaps physical GPU to device 0 in subprocess
            cmd = [
                sys.executable, str(script),
                "--suite", suite, "--gpu", "0",
            ] + extra_args
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            print(f"GPU {gpu}: {args.experiment} --suite {suite}")
            p = subprocess.Popen(cmd, env=env)
            processes.append((gpu, suite, p))

    elif args.split_tasks:
        n_gpus = len(args.gpus)
        tasks_per_gpu = args.n_tasks // n_gpus
        for i, gpu in enumerate(args.gpus):
            start = i * tasks_per_gpu
            end = start + tasks_per_gpu if i < n_gpus - 1 else args.n_tasks
            cmd = [
                sys.executable, str(script),
                "--tasks", *[str(t) for t in range(start, end)],
                "--gpu", "0",
            ] + extra_args
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            print(f"GPU {gpu}: {args.experiment} --tasks {start}-{end-1}")
            p = subprocess.Popen(cmd, env=env)
            processes.append((gpu, f"tasks_{start}_{end}", p))

    else:
        print("Specify --suites or --split-tasks")
        sys.exit(1)

    print(f"\nLaunched {len(processes)} processes. Waiting...")
    for gpu, label, p in processes:
        p.wait()
        status = "OK" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
        print(f"  GPU {gpu} ({label}): {status}")


if __name__ == "__main__":
    main()
