#!/usr/bin/env python3
"""
Bake an OFT result manifest.

Walks ``OFT_DATA_DIR`` once and records, for each (suite, experiment_type) pair,
the most recent timestamped run directory whose ``<run>/<experiment_type>/results.json``
file exists. The manifest is written to ``action_atlas/data/oft_manifest.json``
and is the single source of truth that the API uses at runtime to resolve OFT
result paths (see ``action_atlas/api/experiment_helpers.py``).

Run this script after a new OFT eval batch lands to refresh the manifest.

Usage:

    python scripts/build_oft_manifest.py
    python scripts/build_oft_manifest.py --oft-data-dir /data/openvla_rollouts/openvla_oft
"""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import tyro


@dataclass
class BuildOftManifestConfig:
    # Root of the OFT result tree, expected layout: <oft_data_dir>/<suite>/<run>/<experiment_type>/results.json
    oft_data_dir: str = os.environ.get(
        "OFT_DATA_DIR", "/data/openvla_rollouts/openvla_oft"
    )

    # Where to write the manifest JSON.
    output_path: str = "action_atlas/data/oft_manifest.json"


def discover(oft_data_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Walk the OFT data tree and return a manifest mapping
    suite -> experiment_type -> latest run directory name.
    """
    if not oft_data_dir.exists():
        raise FileNotFoundError(f"OFT data dir not found: {oft_data_dir}")

    manifest: Dict[str, Dict[str, str]] = {}
    for suite_dir in sorted(oft_data_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite_name = suite_dir.name

        suite_entry: Dict[str, str] = {}
        for run_dir in sorted(suite_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            for exp_dir in run_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                exp_type = exp_dir.name
                if (exp_dir / "results.json").exists() and exp_type not in suite_entry:
                    suite_entry[exp_type] = run_name

        if suite_entry:
            manifest[suite_name] = suite_entry
    return manifest


def main(cfg: BuildOftManifestConfig):
    oft_data_dir = Path(cfg.oft_data_dir).resolve()
    output_path = Path(cfg.output_path).resolve()

    manifest = discover(oft_data_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "oft_data_dir": str(oft_data_dir),
        "models": {"openvla": manifest},
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    suites = len(manifest)
    cells = sum(len(v) for v in manifest.values())
    print(f"manifest: suites={suites} cells={cells} -> {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
