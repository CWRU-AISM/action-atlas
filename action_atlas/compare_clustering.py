#!/usr/bin/env python
"""
Compare different clustering and embedding approaches for VLA feature visualization.
Generates comparison images for:
1. TF-IDF + Hierarchical vs TF-IDF + HDBSCAN
2. Sentence-Transformers embeddings + HDBSCAN
3. Concept-based coloring
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
import umap
import hdbscan

# Optional: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Paths
DATA_DIR = Path(__file__).parent / "data"
DESC_DIR = DATA_DIR / "descriptions"
CONCEPT_FILE = DATA_DIR / "concept_features.json"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "clustering_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_descriptions(layer: str = "action_expert_layer_12", suite: str = "concepts"):
    """Load feature descriptions for a layer."""
    desc_file = DESC_DIR / f"descriptions_{layer}_{suite}.json"
    if not desc_file.exists():
        print(f"Description file not found: {desc_file}")
        return None, None

    with open(desc_file) as f:
        data = json.load(f)

    descriptions = data.get("descriptions", {})
    indices = list(descriptions.keys())
    texts = list(descriptions.values())
    return indices, texts


def load_concept_features(layer: str = "action_expert_layer_12"):
    if not CONCEPT_FILE.exists():
        return {}

    with open(CONCEPT_FILE) as f:
        data = json.load(f)

    layer_data = data.get(layer, {})

    # Build feature -> concept mapping
    feature_concepts = {}
    for concept_type in ["motion", "object", "spatial", "action_phase"]:
        for concept_name, concept_data in layer_data.get(concept_type, {}).items():
            for feat_idx in concept_data.get("feature_indices", [])[:50]:  # Top 50 per concept
                if feat_idx not in feature_concepts:
                    feature_concepts[feat_idx] = []
                feature_concepts[feat_idx].append((concept_type, concept_name))

    return feature_concepts


def get_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = min(256, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    return embeddings.astype(np.float32)


def get_sbert_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    if not HAS_SBERT:
        raise ImportError("sentence-transformers not available")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.astype(np.float32)


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    return reducer.fit_transform(embeddings)


def hierarchical_clustering(embeddings, n_clusters=30):
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    return clustering.fit_predict(embeddings)


def hdbscan_clustering(coords, min_cluster_size=15, min_samples=5):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    return clusterer.fit_predict(coords)


def get_concept_colors(indices, feature_concepts):
    """Get colors based on primary concept association."""
    concept_colors = {
        # Motion concepts
        ('motion', 'put'): '#ef4444',
        ('motion', 'open'): '#f97316',
        ('motion', 'push'): '#eab308',
        ('motion', 'interact'): '#84cc16',
        # Object concepts
        ('object', 'bowl'): '#22c55e',
        ('object', 'plate'): '#14b8a6',
        ('object', 'stove'): '#06b6d4',
        ('object', 'cabinet'): '#0ea5e9',
        ('object', 'drawer'): '#3b82f6',
        ('object', 'wine_bottle'): '#6366f1',
        ('object', 'rack'): '#8b5cf6',
        # Spatial concepts
        ('spatial', 'on'): '#a855f7',
        ('spatial', 'in'): '#d946ef',
        ('spatial', 'top'): '#ec4899',
        # Action phase concepts
        ('action_phase', 'approach'): '#f43f5e',
        ('action_phase', 'grasp'): '#fb7185',
        ('action_phase', 'lift'): '#fda4af',
        ('action_phase', 'transport'): '#fecdd3',
        ('action_phase', 'lower'): '#9ca3af',
        ('action_phase', 'release'): '#6b7280',
        ('action_phase', 'retract'): '#4b5563',
    }

    colors = []
    for idx in indices:
        feat_idx = int(idx)
        if feat_idx in feature_concepts and feature_concepts[feat_idx]:
            primary_concept = feature_concepts[feat_idx][0]  # First/primary concept
            color = concept_colors.get(primary_concept, '#cccccc')
        else:
            color = '#cccccc'  # No concept association
        colors.append(color)

    return colors


def generate_cluster_colors(labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Use a colormap
    cmap = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters)))

    colors = []
    for label in labels:
        if label == -1:  # Noise in HDBSCAN
            colors.append('#cccccc')
        else:
            colors.append(plt.matplotlib.colors.rgb2hex(cmap[label % len(cmap)]))

    return colors


def plot_comparison(coords_list, colors_list, titles, output_path, figsize=(16, 5)):
    """Plot comparison of different clustering approaches."""
    n_plots = len(coords_list)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for ax, coords, colors, title in zip(axes, coords_list, colors_list, titles):
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=5, alpha=0.6)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Clustering Comparison for VLA Features")
    print("=" * 60)

    # Load data
    layer = "action_expert_layer_12"
    print(f"\nLoading descriptions for {layer}...")
    indices, texts = load_descriptions(layer)

    if texts is None:
        print("Failed to load descriptions")
        return

    print(f"Loaded {len(texts)} feature descriptions")

    # Load concept associations
    print("Loading concept associations...")
    feature_concepts = load_concept_features(layer)
    print(f"Found concept associations for {len(feature_concepts)} features")

    # 1. TF-IDF embeddings
    print("\n[1] Computing TF-IDF embeddings...")
    tfidf_emb = get_tfidf_embeddings(texts)
    print(f"TF-IDF embedding shape: {tfidf_emb.shape}")

    # 2. UMAP on TF-IDF
    print("[2] Computing UMAP projection...")
    umap_coords = compute_umap(tfidf_emb)

    # 3. Hierarchical clustering
    print("[3] Hierarchical clustering...")
    hier_labels = hierarchical_clustering(tfidf_emb, n_clusters=30)
    hier_colors = generate_cluster_colors(hier_labels)

    # 4. HDBSCAN clustering
    print("[4] HDBSCAN clustering...")
    hdb_labels = hdbscan_clustering(umap_coords, min_cluster_size=10, min_samples=3)
    hdb_colors = generate_cluster_colors(hdb_labels)
    n_hdb_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise = sum(1 for l in hdb_labels if l == -1)
    print(f"HDBSCAN found {n_hdb_clusters} clusters, {n_noise} noise points")

    # 5. Concept-based coloring
    print("[5] Concept-based coloring...")
    concept_colors = get_concept_colors(indices, feature_concepts)

    # Generate comparison plot 1: TF-IDF embeddings
    print("\n[6] Generating comparison plots...")
    plot_comparison(
        [umap_coords, umap_coords, umap_coords],
        [hier_colors, hdb_colors, concept_colors],
        ["TF-IDF + Hierarchical (30 clusters)",
         f"TF-IDF + HDBSCAN ({n_hdb_clusters} clusters)",
         "TF-IDF + Concept Coloring"],
        OUTPUT_DIR / "tfidf_clustering_comparison.png"
    )

    # 6. Sentence-transformers embeddings (if available)
    if HAS_SBERT:
        print("\n[7] Computing Sentence-Transformer embeddings...")
        try:
            sbert_emb = get_sbert_embeddings(texts)
            print(f"SBERT embedding shape: {sbert_emb.shape}")

            # UMAP on SBERT
            print("[8] Computing UMAP on SBERT embeddings...")
            sbert_umap = compute_umap(sbert_emb)

            # HDBSCAN on SBERT UMAP
            print("[9] HDBSCAN on SBERT...")
            sbert_hdb_labels = hdbscan_clustering(sbert_umap, min_cluster_size=10, min_samples=3)
            sbert_hdb_colors = generate_cluster_colors(sbert_hdb_labels)
            n_sbert_clusters = len(set(sbert_hdb_labels)) - (1 if -1 in sbert_hdb_labels else 0)

            # Concept coloring on SBERT UMAP
            print("[10] Generating SBERT comparison...")
            plot_comparison(
                [sbert_umap, sbert_umap],
                [sbert_hdb_colors, concept_colors],
                [f"SBERT + HDBSCAN ({n_sbert_clusters} clusters)",
                 "SBERT + Concept Coloring"],
                OUTPUT_DIR / "sbert_clustering_comparison.png"
            )

            # Compare TF-IDF vs SBERT
            plot_comparison(
                [umap_coords, sbert_umap],
                [concept_colors, concept_colors],
                ["TF-IDF Embeddings (Concept Colors)",
                 "SBERT Embeddings (Concept Colors)"],
                OUTPUT_DIR / "tfidf_vs_sbert_comparison.png"
            )
        except Exception as e:
            print(f"SBERT comparison failed: {e}")

    # Generate legend image
    print("\n[11] Generating concept color legend...")
    generate_legend(OUTPUT_DIR / "concept_color_legend.png")

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


def generate_legend(output_path):
    """Generate a legend showing concept colors."""
    concepts = {
        'Motion': [('put', '#ef4444'), ('open', '#f97316'), ('push', '#eab308'), ('interact', '#84cc16')],
        'Object': [('bowl', '#22c55e'), ('plate', '#14b8a6'), ('stove', '#06b6d4'),
                   ('cabinet', '#0ea5e9'), ('drawer', '#3b82f6'), ('wine_bottle', '#6366f1'), ('rack', '#8b5cf6')],
        'Spatial': [('on', '#a855f7'), ('in', '#d946ef'), ('top', '#ec4899')],
        'Action Phase': [('approach', '#f43f5e'), ('grasp', '#fb7185'), ('lift', '#fda4af'),
                        ('transport', '#fecdd3'), ('lower', '#9ca3af'), ('release', '#6b7280'), ('retract', '#4b5563')],
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    y = 7.5
    for category, items in concepts.items():
        ax.text(0.5, y, category, fontsize=12, fontweight='bold')
        y -= 0.4

        x = 0.5
        for name, color in items:
            ax.scatter([x], [y], c=[color], s=100, marker='s')
            ax.text(x + 0.3, y, name, fontsize=9, va='center')
            x += 2
            if x > 9:
                x = 0.5
                y -= 0.5
        y -= 0.8

    ax.text(0.5, y, 'No Concept', fontsize=10)
    ax.scatter([3.5], [y], c=['#cccccc'], s=100, marker='s')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
