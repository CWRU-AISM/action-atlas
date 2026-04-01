#!/usr/bin/env python
"""
Build Action Atlas data files for VLA SAE features.

This creates:
1. Feature embeddings (using OpenAI text-embedding-3-large)
2. UMAP coordinates and hierarchical clustering
3. Vector database index for similarity search
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

# Optional imports - check availability
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


def get_embeddings(texts: List[str], client=None, batch_size: int = 100, use_tfidf: bool = False, use_sbert: bool = True) -> np.ndarray:
    """Get embeddings for a list of texts.

    Priority:
    1. SBERT (sentence-transformers) - best semantic understanding, local
    2. OpenAI text-embedding-3-large - if API key available
    3. TF-IDF + SVD - fallback if nothing else available

    Args:
        texts: List of text strings to embed
        client: OpenAI client (optional)
        batch_size: Batch size for API calls
        use_tfidf: Force TF-IDF instead of better options
        use_sbert: Use sentence-transformers (default True)
    """
    # Option 1: SBERT (recommended default)
    if use_sbert and HAS_SBERT and not use_tfidf:
        print("Using SBERT embeddings (sentence-transformers)")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        return embeddings.astype(np.float32)

    # Option 2: TF-IDF fallback
    if use_tfidf or client is None:
        print("Using TF-IDF embeddings (fallback)")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Reduce to reasonable dimension
        n_components = min(256, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)

        return embeddings.astype(np.float32)

    # Option 3: OpenAI embeddings
    print("Using OpenAI embeddings")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def compute_umap_coordinates(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.05, spread: float = 0.8) -> np.ndarray:
    """Compute 2D UMAP coordinates from embeddings.

    Args:
        embeddings: Feature embeddings
        n_neighbors: Number of neighbors (higher = more global structure)
        min_dist: Minimum distance between points (lower = tighter clusters)
        spread: How spread out clusters are (lower = tighter)
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric='cosine',
        random_state=42
    )
    coords = reducer.fit_transform(embeddings)
    return coords


def compute_hierarchical_clustering(
    embeddings: np.ndarray,
    n_clusters_list: List[int] = [10, 30, 90]
) -> Dict:
    """Compute hierarchical clustering at multiple levels."""
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    results = {}

    for n_clusters in n_clusters_list:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        results[n_clusters] = labels

    return results


def compute_hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5
) -> np.ndarray:
    """Compute HDBSCAN clustering (density-based).

    HDBSCAN finds natural clusters of varying density without needing
    to specify number of clusters. Points not in any cluster get label -1.
    """
    if not HAS_HDBSCAN:
        raise ImportError("hdbscan is required. Install with: pip install hdbscan")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # Use euclidean on UMAP coords for better density estimation
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


def generate_cluster_colors(n_clusters: int) -> np.ndarray:
    """Generate distinct colors for clusters."""
    np.random.seed(42)
    colors = np.random.rand(n_clusters, 3)
    # Ensure colors are distinguishable
    colors = (colors * 0.7 + 0.15)  # Avoid too dark or too bright
    return colors


def compute_cluster_centers(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute cluster centers in 2D space."""
    unique_labels = np.unique(labels)
    centers = np.zeros((len(unique_labels), 2))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        centers[i] = coords[mask].mean(axis=0)

    return centers


def extract_cluster_topics(
    descriptions: List[str],
    labels: np.ndarray,
    n_words: int = 5
) -> Dict:
    """Extract topic words for each cluster using TF-IDF."""
    if not HAS_SKLEARN:
        return {}

    unique_labels = np.unique(labels)
    topics = {}
    topic_scores = {}

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        feature_names = vectorizer.get_feature_names_out()

        for label in unique_labels:
            mask = labels == label
            cluster_tfidf = tfidf_matrix[mask].mean(axis=0).A1
            top_indices = cluster_tfidf.argsort()[-n_words:][::-1]

            topics[int(label)] = [feature_names[i] for i in top_indices]
            topic_scores[int(label)] = [float(cluster_tfidf[i]) for i in top_indices]

    except Exception as e:
        print(f"Warning: Topic extraction failed: {e}")

    return topics, topic_scores


def build_faiss_index(embeddings: np.ndarray, output_path: Path):
    """Build and save FAISS index for similarity search."""
    if not HAS_FAISS:
        print("Warning: FAISS not available, skipping index creation")
        return

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(output_path / "faiss_index.bin"))
    print(f"Saved FAISS index to {output_path / 'faiss_index.bin'}")


def build_viz_data(
    descriptions_file: Path,
    output_dir: Path,
    suite: str,
    use_tfidf: bool = False
):
    """Build all Action Atlas data files from feature descriptions."""

    # Load descriptions
    with open(descriptions_file) as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to initialize OpenAI client (only used if SBERT not available)
    client = None
    if not use_tfidf and HAS_OPENAI and not HAS_SBERT:
        try:
            client = openai.OpenAI()
            print("OpenAI API available")
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI: {e}")
            # Don't set use_tfidf=True here - let get_embeddings use SBERT as default

    # Process each layer
    for layer_name, layer_data in data.items():
        print(f"\n{'='*60}")
        print(f"Processing {layer_name}")
        print(f"{'='*60}")

        descriptions_dict = layer_data.get('descriptions', {})
        if not descriptions_dict:
            print(f"No descriptions found for {layer_name}")
            continue

        # Convert to lists
        feature_indices = list(descriptions_dict.keys())
        descriptions = list(descriptions_dict.values())

        print(f"Found {len(descriptions)} feature descriptions")

        # Get embeddings
        print("Computing embeddings...")
        embeddings = get_embeddings(descriptions, client, use_tfidf=use_tfidf)

        # Compute UMAP coordinates
        print("Computing UMAP coordinates...")
        coords = compute_umap_coordinates(embeddings)

        # Compute hierarchical clustering
        print("Computing hierarchical clustering...")
        cluster_levels = [10, 30, 90]
        # Adjust cluster levels based on number of features
        cluster_levels = [min(n, len(descriptions) - 1) for n in cluster_levels if n < len(descriptions)]

        clusters = compute_hierarchical_clustering(embeddings, cluster_levels)

        # Build output data
        output_data = {
            'coords': coords,
            'indices': np.array([int(idx) for idx in feature_indices]),
            'descriptions': np.array(descriptions),
        }

        # Add clustering data for each level
        for level in cluster_levels:
            labels = clusters[level]
            n_unique = len(np.unique(labels))
            colors = generate_cluster_colors(n_unique)

            # Assign colors to each point based on label
            point_colors = colors[labels]

            centers = compute_cluster_centers(coords, labels)
            topics, topic_scores = extract_cluster_topics(descriptions, labels)

            output_data[f'cluster_labels_{level}'] = labels
            output_data[f'cluster_colors_{level}'] = point_colors
            output_data[f'cluster_centers_{level}'] = centers
            output_data[f'topic_words_{level}'] = topics
            output_data[f'topic_word_scores_{level}'] = topic_scores

        # Save clustering file
        layer_short = layer_name.replace('action_expert_', '').replace('action_', '')
        clustering_file = output_dir / f"hierarchical_clustering_{layer_short}_{suite}.npz"
        np.savez(clustering_file, **output_data)
        print(f"Saved clustering data to {clustering_file}")

        # Save embeddings file
        embedding_file = output_dir / f"pi05_{layer_name}_{suite}-embedding.npz"
        np.savez(embedding_file, embeddings=embeddings, indices=output_data['indices'], descriptions=descriptions)
        print(f"Saved embeddings to {embedding_file}")

    # Build FAISS index for all embeddings combined (if dimensions match)
    print("\nBuilding FAISS index...")
    # Combine all embeddings
    all_embeddings = []
    all_metadata = []
    embedding_dims = set()

    for layer_name, layer_data in data.items():
        embedding_file = output_dir / f"pi05_{layer_name}_{suite}-embedding.npz"
        if embedding_file.exists():
            emb_data = np.load(embedding_file)
            emb = emb_data['embeddings']
            embedding_dims.add(emb.shape[1])
            all_embeddings.append(emb)
            for idx, desc in zip(emb_data['indices'], emb_data['descriptions']):
                all_metadata.append({
                    'layer': layer_name,
                    'feature_idx': int(idx),
                    'description': str(desc)
                })

    # Save metadata regardless
    with open(output_dir / "feature_metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2)

    if all_embeddings and len(embedding_dims) == 1:
        # All embeddings have same dimension - can combine
        combined_embeddings = np.vstack(all_embeddings)
        build_faiss_index(combined_embeddings, output_dir)
    else:
        print(f"Skipping combined FAISS index (mixed dimensions: {embedding_dims})")
        print("Using TF-IDF embeddings results in variable dimensions per layer")

    print(f"\nAction Atlas data built successfully in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build Action Atlas data for VLA features")
    parser.add_argument('--descriptions', type=str, required=True,
                        help='Path to descriptions JSON file')
    parser.add_argument('--output-dir', type=str, default='action_atlas/data/processed',
                        help='Output directory')
    parser.add_argument('--suite', type=str, default='goal',
                        help='Task suite name')
    parser.add_argument('--use-tfidf', action='store_true',
                        help='Use TF-IDF embeddings instead of OpenAI (no API key required)')
    args = parser.parse_args()

    build_viz_data(
        descriptions_file=Path(args.descriptions),
        output_dir=Path(args.output_dir),
        suite=args.suite,
        use_tfidf=args.use_tfidf
    )


if __name__ == '__main__':
    main()
