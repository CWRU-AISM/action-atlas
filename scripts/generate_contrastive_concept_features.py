#!/usr/bin/env python3
"""
Generate concept_features.json for Action Atlas platform from contrastive concept ID results.

This replaces the old FFN-projection-based concept_features.json with data from the
contrastive concept identification method (Cohen's d x freq scoring).

Data sources:
  Pi0.5:       results/experiment_results/pi05_concept_id/ (108 JSON files)
  OpenVLA-OFT: results/experiment_results/oft_concept_id/  (128 JSON files)

Output:
  action_atlas/data/concept_features.json

The output matches the format expected by the backend's load_concept_features(),
with additional contrastive-specific fields (cohens_d, freq, contrastive_score).
"""

import json
import glob
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PI05_DIR = PROJECT_ROOT / "results" / "experiment_results" / "pi05_concept_id"
OFT_DIR = PROJECT_ROOT / "results" / "experiment_results" / "oft_concept_id"
OUTPUT_FILE = PROJECT_ROOT / "action_atlas" / "data" / "concept_features.json"
BACKUP_FILE = PROJECT_ROOT / "action_atlas" / "data" / "concept_features_ffn_backup.json"


def parse_concept_key(concept_key: str):
    """Parse 'motion/put' -> ('motion', 'put')."""
    parts = concept_key.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "unknown", concept_key


def classify_features_by_strength(cohens_d_values, score_values, freq_values):
    """
    Classify features into strength tiers based on Cohen's d thresholds.

    - 'very_strong': |d| >= 2.0 (very large effect)
    - 'strong':      |d| >= 0.8 (large effect)
    - 'concept':     |d| >= 0.2 (small-to-medium effect, i.e., all meaningful)

    Returns indices (within the top-N list) for each tier.
    """
    concept_indices = []
    strong_indices = []
    very_strong_indices = []

    for i, d_val in enumerate(cohens_d_values):
        abs_d = abs(d_val)
        if abs_d >= 0.2:
            concept_indices.append(i)
        if abs_d >= 0.8:
            strong_indices.append(i)
        if abs_d >= 2.0:
            very_strong_indices.append(i)

    return concept_indices, strong_indices, very_strong_indices


def process_contrastive_file(filepath: str):
    """
    Process a single contrastive concept ID JSON file.
    Returns a dict: { (concept_type, concept_name): concept_platform_entry }
    """
    with open(filepath) as f:
        data = json.load(f)

    results = {}
    n_features_total = None

    for concept_key, concept_data in data.get("concepts", {}).items():
        concept_type, concept_name = parse_concept_key(concept_key)

        top_features = concept_data.get("top_features", [])
        top_scores = concept_data.get("top_scores", [])
        top_cohens_d = concept_data.get("top_cohens_d", [])
        top_freq = concept_data.get("top_freq", [])
        n_features = concept_data.get("n_features", 0)

        if n_features_total is None:
            n_features_total = n_features

        # Classify features by Cohen's d strength
        concept_idx, strong_idx, very_strong_idx = classify_features_by_strength(
            top_cohens_d, top_scores, top_freq
        )

        # Map back to actual feature indices
        feature_indices = [top_features[i] for i in concept_idx]
        strong_feature_indices = [top_features[i] for i in strong_idx]
        very_strong_feature_indices = [top_features[i] for i in very_strong_idx]

        # Top 20 for visualization (indices and their contrastive scores)
        top_20_indices = top_features[:20]
        top_20_scores = top_scores[:20]
        # Normalize scores to [0, 1] range for ratios (relative to max)
        max_score = top_20_scores[0] if top_20_scores else 1.0
        top_20_ratios = [s / max_score if max_score > 0 else 0.0 for s in top_20_scores]

        entry = {
            # Core fields expected by backend
            "concept_features": len(feature_indices),
            "total_features": n_features,
            "strong_features": len(strong_feature_indices),
            "very_strong_features": len(very_strong_feature_indices),
            "feature_indices": feature_indices[:100],  # cap at 100
            "strong_feature_indices": strong_feature_indices[:100],
            "very_strong_feature_indices": very_strong_feature_indices[:100],
            "top_20_indices": top_20_indices,
            "top_20_ratios": top_20_ratios,
            # Contrastive-specific fields
            "top_50_indices": top_features[:50],
            "top_50_scores": top_scores[:50],
            "top_50_cohens_d": top_cohens_d[:50],
            "top_50_freq": top_freq[:50] if top_freq else [],
            "mean_cohens_d": concept_data.get("mean_cohens_d", 0.0),
            "max_cohens_d": concept_data.get("max_cohens_d", 0.0),
            "n_features_d_gt_1": concept_data.get("n_features_d_gt_1", 0),
            "n_features_d_gt_2": concept_data.get("n_features_d_gt_2", 0),
            "n_concept_samples": concept_data.get("n_concept_samples", 0),
            "n_other_samples": concept_data.get("n_other_samples", 0),
        }

        results[(concept_type, concept_name)] = entry

    return results, n_features_total


def build_layer_key_pi05(pathway: str, layer_num: int) -> str:
    """
    Build the platform layer key for Pi0.5.

    Old format used: 'action_expert_layer_N'
    New format:      'action_expert_layer_N'  (expert) or 'paligemma_layer_N' (paligemma)
    """
    if pathway == "expert":
        return f"action_expert_layer_{layer_num}"
    elif pathway == "paligemma":
        return f"paligemma_layer_{layer_num}"
    else:
        return f"{pathway}_layer_{layer_num}"


def build_layer_key_oft(layer_num: int) -> str:
    """Build platform layer key for OpenVLA-OFT."""
    return f"openvla_oft_layer_{layer_num}"


def process_pi05_files():
    """
    Process all Pi0.5 contrastive concept ID files.

    File naming: pi05_{pathway}_concept_id_layer{NN}_{suite}.json
    where pathway = expert | paligemma
    and suite = goal | object | spatial | 10
    """
    pattern = str(PI05_DIR / "pi05_*_concept_id_layer*_*.json")
    files = sorted(glob.glob(pattern))

    # Aggregate: layer_key -> concept_type -> concept_name -> entry
    # We merge across suites for the same layer (concepts from different suites
    # may differ; we keep the best score for shared concepts and add unique ones)
    layer_data = defaultdict(lambda: defaultdict(dict))
    layer_suites = defaultdict(set)

    file_regex = re.compile(
        r"pi05_(expert|paligemma)_concept_id_layer(\d+)_(\w+)\.json"
    )

    for filepath in files:
        basename = os.path.basename(filepath)
        match = file_regex.match(basename)
        if not match:
            print(f"  WARNING: Skipping unrecognized file: {basename}")
            continue

        pathway = match.group(1)
        layer_num = int(match.group(2))
        suite = match.group(3)

        layer_key = build_layer_key_pi05(pathway, layer_num)
        layer_suites[layer_key].add(suite)

        concepts, n_features = process_contrastive_file(filepath)

        for (concept_type, concept_name), entry in concepts.items():
            existing = layer_data[layer_key][concept_type].get(concept_name)
            if existing is None:
                # New concept for this layer - add it, tag with source suite
                entry["source_suites"] = [suite]
                layer_data[layer_key][concept_type][concept_name] = entry
            else:
                # Concept seen from another suite - merge by keeping the one
                # with the higher max Cohen's d (stronger signal)
                existing["source_suites"].append(suite)
                if entry["max_cohens_d"] > existing["max_cohens_d"]:
                    # Replace with the stronger version but keep merged suites
                    suites_list = existing["source_suites"]
                    entry["source_suites"] = suites_list
                    layer_data[layer_key][concept_type][concept_name] = entry

    print(f"  Pi0.5: Processed {len(files)} files across {len(layer_data)} layers")
    for lk in sorted(layer_data.keys()):
        n_concepts = sum(len(v) for v in layer_data[lk].values())
        suites = sorted(layer_suites[lk])
        print(f"    {lk}: {n_concepts} concepts from suites {suites}")

    return dict(layer_data)


def process_oft_files():
    """
    Process all OpenVLA-OFT contrastive concept ID files.

    File naming: oft_concept_id_layer{NN}_libero_{suite}.json
    """
    pattern = str(OFT_DIR / "oft_concept_id_layer*_libero_*.json")
    files = sorted(glob.glob(pattern))

    layer_data = defaultdict(lambda: defaultdict(dict))
    layer_suites = defaultdict(set)

    file_regex = re.compile(
        r"oft_concept_id_layer(\d+)_libero_(\w+)\.json"
    )

    for filepath in files:
        basename = os.path.basename(filepath)
        match = file_regex.match(basename)
        if not match:
            print(f"  WARNING: Skipping unrecognized file: {basename}")
            continue

        layer_num = int(match.group(1))
        suite = match.group(2)

        layer_key = build_layer_key_oft(layer_num)
        layer_suites[layer_key].add(suite)

        concepts, n_features = process_contrastive_file(filepath)

        for (concept_type, concept_name), entry in concepts.items():
            existing = layer_data[layer_key][concept_type].get(concept_name)
            if existing is None:
                entry["source_suites"] = [suite]
                layer_data[layer_key][concept_type][concept_name] = entry
            else:
                existing["source_suites"].append(suite)
                if entry["max_cohens_d"] > existing["max_cohens_d"]:
                    suites_list = existing["source_suites"]
                    entry["source_suites"] = suites_list
                    layer_data[layer_key][concept_type][concept_name] = entry

    print(f"  OFT: Processed {len(files)} files across {len(layer_data)} layers")
    for lk in sorted(layer_data.keys(), key=lambda x: int(x.split('_')[-1])):
        n_concepts = sum(len(v) for v in layer_data[lk].values())
        suites = sorted(layer_suites[lk])
        print(f"    {lk}: {n_concepts} concepts from suites {suites}")

    return dict(layer_data)


def build_output(pi05_data: dict, oft_data: dict) -> dict:
    """
    Build the final output dict. Structure:

    {
        "_metadata": { ... },
        "action_expert_layer_0": { "motion": { "put": { ... } }, ... },
        "paligemma_layer_0": { ... },
        "openvla_oft_layer_0": { ... },
        ...
    }
    """
    output = {}

    # Metadata block
    output["_metadata"] = {
        "concept_method": "contrastive",
        "scoring_formula": "score_f = cohens_d_f * freq_f",
        "strength_thresholds": {
            "concept": "abs(d) >= 0.2",
            "strong": "abs(d) >= 0.8",
            "very_strong": "abs(d) >= 2.0"
        },
        "models": {
            "pi05": {
                "pathways": ["action_expert", "paligemma"],
                "n_layers_per_pathway": 18,
                "sae_width_expert": 8192,
                "sae_width_paligemma": 16384,
                "source_dir": "results/experiment_results/pi05_concept_id/"
            },
            "openvla_oft": {
                "n_layers": 32,
                "sae_width": 32768,
                "source_dir": "results/experiment_results/oft_concept_id/"
            }
        },
        "generated_at": datetime.now().isoformat(),
        "previous_method": "ffn_projection (backed up to concept_features_ffn_backup.json)"
    }

    # Add Pi0.5 layers
    for layer_key, layer_content in pi05_data.items():
        output[layer_key] = dict(layer_content)

    # Add OFT layers
    for layer_key, layer_content in oft_data.items():
        output[layer_key] = dict(layer_content)

    return output


def print_summary(output: dict):
    """Print a summary of the generated data."""
    metadata = output.get("_metadata", {})

    # Count by model
    pi05_expert_layers = [k for k in output if k.startswith("action_expert_layer_")]
    pi05_pali_layers = [k for k in output if k.startswith("paligemma_layer_")]
    oft_layers = [k for k in output if k.startswith("openvla_oft_layer_")]

    total_concepts_pi05 = 0
    total_concepts_oft = 0
    all_concept_types = set()
    all_concepts = set()

    for layer_key in pi05_expert_layers + pi05_pali_layers:
        for ct, concepts in output[layer_key].items():
            if isinstance(concepts, dict):
                total_concepts_pi05 += len(concepts)
                all_concept_types.add(ct)
                for cn in concepts:
                    all_concepts.add(f"{ct}/{cn}")

    for layer_key in oft_layers:
        for ct, concepts in output[layer_key].items():
            if isinstance(concepts, dict):
                total_concepts_oft += len(concepts)
                all_concept_types.add(ct)
                for cn in concepts:
                    all_concepts.add(f"{ct}/{cn}")

    print(f"\n=== SUMMARY ===")
    print(f"  Method: {metadata.get('concept_method', 'unknown')}")
    print(f"  Pi0.5 Expert layers: {len(pi05_expert_layers)}")
    print(f"  Pi0.5 PaliGemma layers: {len(pi05_pali_layers)}")
    print(f"  OpenVLA-OFT layers: {len(oft_layers)}")
    print(f"  Pi0.5 total concept entries: {total_concepts_pi05}")
    print(f"  OFT total concept entries: {total_concepts_oft}")
    print(f"  Unique concept types: {sorted(all_concept_types)}")
    print(f"  Unique concepts: {len(all_concepts)}")
    print(f"  All concepts: {sorted(all_concepts)}")


def main():
    print("Generating concept_features.json from contrastive concept ID data...")
    print(f"  Pi0.5 source: {PI05_DIR}")
    print(f"  OFT source:   {OFT_DIR}")
    print(f"  Output:        {OUTPUT_FILE}")
    print()

    # Check source directories exist
    if not PI05_DIR.exists():
        print(f"ERROR: Pi0.5 directory not found: {PI05_DIR}")
        return
    if not OFT_DIR.exists():
        print(f"ERROR: OFT directory not found: {OFT_DIR}")
        return

    # Backup existing file
    if OUTPUT_FILE.exists():
        print(f"  Backing up existing concept_features.json -> {BACKUP_FILE.name}")
        import shutil
        shutil.copy2(OUTPUT_FILE, BACKUP_FILE)

    # Process all files
    print("\nProcessing Pi0.5 contrastive concept ID files...")
    pi05_data = process_pi05_files()

    print("\nProcessing OpenVLA-OFT contrastive concept ID files...")
    oft_data = process_oft_files()

    # Build and write output
    output = build_output(pi05_data, oft_data)
    print_summary(output)

    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Written: {file_size_mb:.1f} MB")

    # Validate: check that the backend can read the new format
    print("\nValidation: checking key fields expected by backend...")
    errors = []
    for layer_key, layer_data in output.items():
        if layer_key.startswith("_"):
            continue
        for concept_type, concepts in layer_data.items():
            if not isinstance(concepts, dict):
                continue
            for concept_name, entry in concepts.items():
                required = ["concept_features", "total_features", "strong_features",
                           "very_strong_features", "feature_indices", "top_20_indices",
                           "top_20_ratios"]
                for field in required:
                    if field not in entry:
                        errors.append(f"  Missing '{field}' in {layer_key}/{concept_type}/{concept_name}")

    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(e)
    else:
        print("  All entries have required fields. OK.")

    # Check UMAP compatibility
    print("\nUMAP Status:")
    processed_dir = PROJECT_ROOT / "action_atlas" / "data" / "processed"
    pi05_umap_files = list(processed_dir.glob("pi05_*embedding.npz"))
    oft_umap_files = list(processed_dir.glob("openvla/*.npz")) if (processed_dir / "openvla").exists() else []
    hierarchical_files = list(processed_dir.glob("hierarchical_*.npz"))
    print(f"  Pi0.5 UMAP NPZ files: {len(pi05_umap_files)}")
    print(f"  OFT UMAP NPZ files: {len(oft_umap_files)}")
    print(f"  Hierarchical clustering NPZ files: {len(hierarchical_files)}")
    print(f"  NOTE: Existing UMAPs use FFN projection features. They will need")
    print(f"  regeneration to match contrastive concept assignments.")
    print(f"  The UMAP embeddings themselves (feature activations) are model-derived")
    print(f"  and not method-dependent, but concept COLOR LABELS in the visualization")
    print(f"  should be updated to use contrastive concept assignments.")

    # Check description coverage
    print("\nDescription Coverage:")
    desc_dir = PROJECT_ROOT / "action_atlas" / "data" / "descriptions"
    if desc_dir.exists():
        desc_files = list(desc_dir.glob("*.json"))
        print(f"  Description JSON files: {len(desc_files)}")
        # Check if descriptions cover the contrastive feature indices
        try:
            all_desc = json.load(open(desc_dir / "all_descriptions_concepts.json"))
            pi05_desc_layers = list(all_desc.keys())
            print(f"  Pi0.5 description layers (concepts): {len(pi05_desc_layers)}")
            # Count total described features
            total_described = sum(
                len(v.get("descriptions", {})) for v in all_desc.values()
            )
            print(f"  Total Pi0.5 described features (concepts): {total_described}")
        except Exception as e:
            print(f"  Could not read descriptions: {e}")

        print(f"  NOTE: Descriptions are per-feature (not per-concept), so they remain")
        print(f"  valid regardless of concept identification method. The same SAE features")
        print(f"  are described; only the concept-to-feature MAPPING changes.")

    print("\nDone.")


if __name__ == "__main__":
    main()
