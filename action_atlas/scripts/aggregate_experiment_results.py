#!/usr/bin/env python3
"""
Aggregate experiment results from SmolVLA, X-VLA, and GR00T into unified JSON files
for the Action Atlas platform.

Reads raw experiment data (baselines, grid ablation, counterfactual prompting,
cross-task injection, vision perturbation, displacement analysis, fraction-to-failure,
steering, temporal ablation) and produces one JSON per model:
    action_atlas/data/experiment_results_{model}.json

Usage:
    python aggregate_experiment_results.py [--models smolvla xvla groot] [--output-dir PATH]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
# Helpers
def safe_load_json(path: Path) -> dict | None:
    # Load a JSON file, returning None on failure
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, FileNotFoundError):
        return None


def load_jsonl(path: Path) -> list[dict]:
    # Load a JSONL file, returning a list of dicts
    results = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except (IOError, FileNotFoundError):
        pass
    return results


def compute_success_rate(episodes: list[dict]) -> float:
    # Compute success rate from a list of episode dicts with 'success' key
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if ep.get("success", False)) / len(episodes)


def round_val(v, decimals=4):
    # Round a numeric value for JSON output
    if isinstance(v, float):
        return round(v, decimals)
    return v
# SmolVLA aggregation
SMOLVLA_LIBERO_DIR = Path("/data/smolvla_libero")
SMOLVLA_METAWORLD_DIR = Path("/data/smolvla_rollouts")
SMOLVLA_LIBERO_SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]
SMOLVLA_MW_DIFFICULTIES = ["easy", "medium", "hard", "very_hard"]



from action_atlas.scripts.aggregate_smolvla import *
from action_atlas.scripts.aggregate_xvla import *
from action_atlas.scripts.aggregate_groot import *
