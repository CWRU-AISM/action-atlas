'''
Download Action Atlas release data from HuggingFace, with subset selection.

Two kinds of data back this project. The visualization bundle is what the web frontend reads
(clustering layouts, feature embeddings, concept descriptions); it is small and is all most users
need to run the site locally. The reproduction data is heavier (SAE weights, identified concepts,
trajectories, and raw activations) and is only needed to retrain SAEs or rerun experiments.

Everything lands at the relative paths the app and the experiment scripts already expect, so no code
or path edits are required after downloading. You can fetch only what you need: a single model, a
single SAE pooling arm, specific LIBERO suites, or just the visualization bundle.

Examples:
    python setup/download_data.py --viz
    python setup/download_data.py --models pi05 --types saes --arms per_token
    python setup/download_data.py --models oft --types activations --suites libero_goal
    python setup/download_data.py --models all --types saes,concepts,trajectories
    python setup/download_data.py --list
'''
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = ["oft", "pi05", "xvla", "groot", "smolvla"]
ARMS = ["per_token", "mean_pool", "temporal"]
BUNDLE_TYPES = ["saes", "concepts", "trajectories"]

# The HuggingFace dataset repos. Edit here if the published names change.
VIZ_REPO = "bag100/action-atlas-viz"
BUNDLE_REPO = "bag100/action-atlas-{model}"
# Activation repos. X-VLA is split by environment plus its SAE-training caches.
ACTIVATION_REPOS = {
    "oft": "bag100/action-atlas-oft-activations",
    "pi05": "bag100/action-atlas-pi05-activations",
    "xvla": {"libero": "bag100/action-atlas-xvla-libero-activations",
             "simplerenv": "bag100/action-atlas-xvla-simplerenv-activations",
             "saetrain": "bag100/action-atlas-xvla-saetrain"},
    "groot": "bag100/action-atlas-groot-activations",
    "smolvla": "bag100/action-atlas-smolvla-activations",
}


def bundle_patterns(types, arms):
    '''
    Build allow_patterns for the per-model bundle repo from the requested artifact types and SAE arms.
    The manifest, loader, and readme are always included so the download is self-describing.
    '''
    patterns = ["manifest.jsonl", "README.md", "loader.py"]
    if "saes" in types:
        for arm in arms:
            patterns.append(f"saes/{arm}/**")
    if "concepts" in types:
        patterns.append("concepts/**")
    if "trajectories" in types:
        patterns.append("trajectories/**")
    return patterns


def fetch(repo_id, dest, allow_patterns=None, force=False):
    # Snapshot a dataset repo (or a subset of it) into dest; skip when already present.
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()) and not force:
        print(f"{dest} already populated; use --force to re-download")
        return dest
    print(f"downloading {repo_id} -> {dest}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest),
                      allow_patterns=allow_patterns)
    return dest


def download_viz(out, force):
    # The frontend bundle lands under action_atlas/data so the site reads it without configuration.
    print("Visualization bundle (frontend):")
    fetch(VIZ_REPO, out / "action_atlas" / "data", force=force)


def download_bundle(model, types, arms, out, force):
    print(f"Bundle for {model} ({', '.join(types)}):")
    fetch(BUNDLE_REPO.format(model=model), out / "outputs" / "release" / model,
          allow_patterns=bundle_patterns(types, arms), force=force)


def download_activations(model, suites, out, force):
    print(f"Activations for {model} (suites: {', '.join(suites) if suites else 'all'}):")
    spec = ACTIVATION_REPOS[model]
    dest_root = out / "activations" / model
    if isinstance(spec, dict):
        targets = spec.items() if not suites else [(s, spec[s]) for s in suites if s in spec]
        for suite, repo_id in targets:
            fetch(repo_id, dest_root / suite, force=force)
    else:
        patterns = [f"**/{s}/**" for s in suites] if suites else None
        fetch(spec, dest_root, allow_patterns=patterns, force=force)


def list_release():
    print("Action Atlas release data\n")
    print("Visualization bundle (frontend): --viz")
    print(f"repo {VIZ_REPO} -> action_atlas/data/  (about 0.4 GB)\n")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Artifact types: viz, {', '.join(BUNDLE_TYPES)}, activations")
    print(f"SAE pooling arms: {', '.join(ARMS)} (per_token is the primary release)\n")
    print("Per-model bundle repos (SAEs + concepts + trajectories):")
    for m in MODELS:
        print(f"{BUNDLE_REPO.format(model=m)} -> outputs/release/{m}/")
    print("\nActivation repos (heavy, only for reproduction):")
    for m, spec in ACTIVATION_REPOS.items():
        first = next(iter(spec.values())) if isinstance(spec, dict) else spec
        print(f"{m}: {first}{' (per suite)' if isinstance(spec, dict) else ''} -> activations/{m}/")


def parse_csv(value, allowed, label):
    if value == "all":
        return list(allowed)
    items = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise SystemExit(f"unknown {label}: {bad}; choose from {allowed} or 'all'")
    return items


def main():
    parser = argparse.ArgumentParser(description="Download Action Atlas release data (subset-aware)")
    parser.add_argument("--viz", action="store_true",
                        help="download the visualization bundle the frontend needs")
    parser.add_argument("--models", default=None,
                        help=f"comma list of {MODELS} or 'all'")
    parser.add_argument("--types", default="saes",
                        help="comma list of: saes, concepts, trajectories, activations")
    parser.add_argument("--arms", default="per_token",
                        help=f"comma list of SAE pooling arms {ARMS} or 'all'")
    parser.add_argument("--suites", default=None,
                        help="comma list of LIBERO suites to limit activations (default: all)")
    parser.add_argument("--output", default=".", help="repo root to download into")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--list", action="store_true", help="list the release and exit")
    args = parser.parse_args()

    if args.list or (not args.viz and not args.models):
        list_release()
        if not args.viz and not args.models:
            print("\nNothing selected. Pass --viz and/or --models. See --help.")
        return

    out = Path(args.output)
    if args.viz:
        download_viz(out, args.force)
    if args.models:
        models = parse_csv(args.models, MODELS, "model")
        types = parse_csv(args.types, BUNDLE_TYPES + ["activations"], "type")
        arms = parse_csv(args.arms, ARMS, "arm")
        suites = args.suites.split(",") if args.suites else None
        bundle_types = [t for t in types if t in BUNDLE_TYPES]
        for model in models:
            if bundle_types:
                download_bundle(model, bundle_types, arms, out, args.force)
            if "activations" in types:
                download_activations(model, suites, out, args.force)
    print("\nDone.")


if __name__ == "__main__":
    main()
