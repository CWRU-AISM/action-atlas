#!/usr/bin/env python
"""
Generate clustering NPZ files for all models that need them.
Handles GR00T and SmolVLA aggregate description files.
"""
import json
import numpy as np
from pathlib import Path
from build_viz_data import (
    get_embeddings,
    compute_umap_coordinates,
    compute_hierarchical_clustering,
    generate_cluster_colors,
    compute_cluster_centers,
    extract_cluster_topics,
    build_faiss_index,
)

DESC_DIR = Path("data/descriptions/contrastive")
OUT_DIR = Path("data/processed/contrastive")


def process_aggregate_file(agg_path: Path, model: str, output_dir: Path):
    """Process one aggregate description file → one clustering NPZ per layer."""
    with open(agg_path) as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_embeddings = []
    all_metadata = []

    for layer_key, layer_data in data.items():
        descs_dict = layer_data.get("descriptions", {})
        if not descs_dict or len(descs_dict) < 3:
            print(f"  Skipping {layer_key}: only {len(descs_dict)} descriptions")
            continue

        feature_indices = list(descs_dict.keys())
        descriptions = list(descs_dict.values())
        print(f"  {layer_key}: {len(descriptions)} features")

        # Embeddings
        embeddings = get_embeddings(descriptions, use_sbert=True)

        # UMAP
        n_neighbors = min(15, len(descriptions) - 1)
        coords = compute_umap_coordinates(embeddings, n_neighbors=n_neighbors)

        # Hierarchical clustering
        max_n = len(descriptions) - 1
        cluster_levels = [n for n in [10, 30, 90] if n < max_n]
        if not cluster_levels:
            cluster_levels = [max(2, max_n // 2)]

        clusters = compute_hierarchical_clustering(embeddings, cluster_levels)

        # Build output
        output_data = {
            "coords": coords,
            "indices": np.array([int(idx) for idx in feature_indices]),
            "descriptions": np.array(descriptions),
        }
        for level in cluster_levels:
            labels = clusters[level]
            n_unique = len(np.unique(labels))
            colors = generate_cluster_colors(n_unique)
            point_colors = colors[labels]
            centers = compute_cluster_centers(coords, labels)
            topics, topic_scores = extract_cluster_topics(descriptions, labels)
            output_data[f"cluster_labels_{level}"] = labels
            output_data[f"cluster_colors_{level}"] = point_colors
            output_data[f"cluster_centers_{level}"] = centers
            output_data[f"topic_words_{level}"] = topics
            output_data[f"topic_word_scores_{level}"] = topic_scores

        # Strip suite from layer_key for filename
        # GR00T: "dit_L00_libero_goal" → layer_short="dit_L00", suite="libero_goal"
        # SmolVLA: "layer_0_libero_goal" → layer_short="layer_0", suite="libero_goal"
        parts = layer_key.split("_")
        # Find where libero starts
        suite_start = None
        for i, p in enumerate(parts):
            if p == "libero":
                suite_start = i
                break
        if suite_start is not None:
            layer_short = "_".join(parts[:suite_start])
            suite = "_".join(parts[suite_start:])
        else:
            layer_short = layer_key
            suite = "unknown"

        # For SmolVLA expert, prefix with "expert_" if not already
        if model == "smolvla" and "expert" in str(agg_path) and not layer_short.startswith("expert"):
            layer_short = f"expert_{layer_short}"
        if model == "smolvla" and "vlm" in str(agg_path) and not layer_short.startswith("vlm"):
            layer_short = f"vlm_{layer_short}"

        clustering_file = output_dir / f"hierarchical_clustering_{layer_short}_{suite}.npz"
        np.savez(clustering_file, **output_data)

        embedding_file = output_dir / f"{model}_{layer_short}_{suite}-embedding.npz"
        np.savez(embedding_file, embeddings=embeddings, indices=output_data["indices"], descriptions=descriptions)

        all_embeddings.append(embeddings)
        for idx, desc in zip(feature_indices, descriptions):
            all_metadata.append({"layer": layer_key, "feature_idx": int(idx), "description": desc})

    # FAISS index
    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    if all_embeddings:
        dims = set(e.shape[1] for e in all_embeddings)
        if len(dims) == 1:
            combined = np.vstack(all_embeddings)
            build_faiss_index(combined, output_dir)

    print(f"  Done: {output_dir}")


def main():
    # ===== GR00T =====
    groot_desc_dir = DESC_DIR / "groot"
    groot_out_dir = OUT_DIR / "groot"
    groot_agg_files = sorted(groot_desc_dir.glob("all_descriptions_groot_*.json"))
    print(f"\n{'='*60}")
    print(f"GR00T: {len(groot_agg_files)} aggregate files")
    print(f"{'='*60}")
    for agg in groot_agg_files:
        print(f"\nProcessing {agg.name}...")
        process_aggregate_file(agg, "groot", groot_out_dir)

    # ===== SmolVLA =====
    smolvla_desc_dir = DESC_DIR / "smolvla"
    smolvla_out_dir = OUT_DIR / "smolvla"
    smolvla_agg_files = sorted(smolvla_desc_dir.glob("all_descriptions_smolvla_*.json"))
    print(f"\n{'='*60}")
    print(f"SmolVLA: {len(smolvla_agg_files)} aggregate files")
    print(f"{'='*60}")
    for agg in smolvla_agg_files:
        print(f"\nProcessing {agg.name}...")
        process_aggregate_file(agg, "smolvla", smolvla_out_dir)

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_dir in [groot_out_dir, smolvla_out_dir]:
        if model_dir.exists():
            npz_files = list(model_dir.glob("hierarchical_clustering_*.npz"))
            emb_files = list(model_dir.glob("*-embedding.npz"))
            print(f"  {model_dir.name}: {len(npz_files)} clustering + {len(emb_files)} embedding files")


if __name__ == "__main__":
    main()
