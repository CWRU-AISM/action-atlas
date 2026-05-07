#!/usr/bin/env python3
"""
Generate publication-quality UMAP visualizations.
- Light background (paper-friendly)
- Large, visible points with edge outlines
- Zoomed view focusing on cluster structure
- Vibrant but print-safe colors
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Vibrant colors that work on white background
CLUSTER_COLORS = [
    '#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261',
    '#264653', '#A8DADC', '#1D3557', '#F77F00', '#D62828',
    '#023E8A', '#0077B6', '#0096C7', '#00B4D8', '#48CAE4',
    '#90E0EF', '#003566', '#FFC300', '#E07A5F', '#3D405B',
    '#81B29A', '#F2CC8F', '#6D597A', '#B56576', '#E56B6F',
    '#EAAC8B', '#355070', '#6D597A', '#B56576', '#E88C7D'
]

LAYER_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#1D3557", "#F77F00", "#D62828", "#023E8A",
    "#0077B6", "#0096C7", "#00B4D8", "#48CAE4", "#90E0EF",
    "#003566", "#FFC300", "#E07A5F"
]


def load_layer_data(layer_num: int):
    # Load UMAP data for a specific layer
    possible_files = [
        DATA_DIR / f"hierarchical_clustering_layer_{layer_num}_goal.npz",
        DATA_DIR / f"hierarchical_clustering_layer_{layer_num}_concepts.npz",
    ]

    for npz_path in possible_files:
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            result = {
                'coords': data['coords'],
                'descriptions': data['descriptions'] if 'descriptions' in data else None,
                'cluster_labels_30': data['cluster_labels_30'] if 'cluster_labels_30' in data else None,
                'cluster_centers_30': data['cluster_centers_30'] if 'cluster_centers_30' in data else None,
                'topic_words_30': data['topic_words_30'].item() if 'topic_words_30' in data else None,
            }
            return result
    return None


def get_cluster_label(topic_words: list, max_words: int = 2) -> str:
    # Extract a clean, short label from topic words
    if not topic_words:
        return ""
    # Take first few words and clean them up
    label_words = []
    for word in topic_words[:max_words]:
        # Clean up compound phrases
        word = word.strip()
        if len(word) > 15:
            word = word.split()[0] if ' ' in word else word[:12]
        label_words.append(word)
    return '\n'.join(label_words)


def plot_layer_large(layer_num: int, dpi: int = 300, point_size: int = 50, add_labels: bool = True):
    # Generate clean, large-point UMAP for paper with concept labels
    data = load_layer_data(layer_num)
    if data is None:
        print(f"No data for layer {layer_num}")
        return None

    coords = data['coords']
    cluster_labels = data['cluster_labels_30']
    cluster_centers = data.get('cluster_centers_30')
    topic_words = data.get('topic_words_30')

    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)

    if cluster_labels is not None:
        unique_labels = np.unique(cluster_labels)
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, s=point_size, alpha=0.85,
                edgecolors='white', linewidth=0.5,
                label=f'C{label}' if i < 10 else None
            )

        # Add text labels for clusters
        if add_labels and topic_words and cluster_centers is not None:
            # Get cluster sizes to label only larger clusters
            unique, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))

            # Label top N clusters by size
            top_n = min(15, len(unique_labels))
            top_clusters = sorted(cluster_sizes.keys(), key=lambda k: cluster_sizes[k], reverse=True)[:top_n]

            for label in top_clusters:
                if label in topic_words and label < len(cluster_centers):
                    words = topic_words[label]
                    label_text = get_cluster_label(words, max_words=2)
                    center = cluster_centers[label]

                    # Add text with background box for readability
                    ax.annotate(
                        label_text,
                        xy=(center[0], center[1]),
                        fontsize=8,
                        fontweight='bold',
                        ha='center', va='center',
                        color='black',
                        bbox=dict(
                            boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='gray',
                            alpha=0.85
                        )
                    )
    else:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=LAYER_COLORS[layer_num], s=point_size, alpha=0.85,
            edgecolors='white', linewidth=0.5
        )

    # Zoom to data bounds with padding
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad = 0.08 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(f'Layer {layer_num} SAE Features\n({len(coords):,} features, 30 clusters)',
                fontsize=16, fontweight='bold', pad=15)

    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)

    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"umap_layer_{layer_num}_paper.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()
    return output_path


def plot_all_layers_paper(dpi: int = 300):
    # Combined plot with all layers - paper quality
    fig, ax = plt.subplots(figsize=(14, 12), dpi=dpi)

    all_coords = []
    all_colors = []
    layer_counts = []

    for layer_num in range(18):
        data = load_layer_data(layer_num)
        if data is None:
            continue

        coords = data['coords']
        color = LAYER_COLORS[layer_num]

        all_coords.append(coords)
        all_colors.extend([color] * len(coords))
        layer_counts.append((layer_num, len(coords)))

    if not all_coords:
        print("No data found")
        return None

    all_coords = np.vstack(all_coords)

    ax.scatter(
        all_coords[:, 0], all_coords[:, 1],
        c=all_colors, s=15, alpha=0.6, edgecolors='none'
    )

    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(f'SAE Features Across All Layers\n({len(all_coords):,} features)',
                fontsize=18, fontweight='bold', pad=15)

    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle='-')

    # Legend
    handles = []
    for layer_num, count in layer_counts:
        handle = plt.Line2D([0], [0], marker='o', color='white',
                           markerfacecolor=LAYER_COLORS[layer_num], markersize=10,
                           markeredgecolor='gray', markeredgewidth=0.5,
                           label=f'Layer {layer_num} ({count:,})')
        handles.append(handle)

    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
             fontsize=9, framealpha=0.95, ncol=1, edgecolor='gray')

    plt.tight_layout()

    output_path = OUTPUT_DIR / "umap_all_layers_paper.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    return output_path


def plot_layer_grid_paper(dpi: int = 300, add_labels: bool = True):
    # 3x6 grid for paper - compact but visible with concept labels
    fig, axes = plt.subplots(3, 6, figsize=(28, 14), dpi=dpi)
    axes = axes.flatten()

    for layer_num in range(18):
        ax = axes[layer_num]
        data = load_layer_data(layer_num)

        if data is None:
            ax.text(0.5, 0.5, f'L{layer_num}\nNo data',
                   ha='center', va='center', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            coords = data['coords']
            cluster_labels = data['cluster_labels_30']
            cluster_centers = data.get('cluster_centers_30')
            topic_words = data.get('topic_words_30')

            if cluster_labels is not None:
                unique_labels = np.unique(cluster_labels)
                for i, label in enumerate(unique_labels):
                    mask = cluster_labels == label
                    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                              c=color, s=25, alpha=0.8, edgecolors='white', linewidth=0.3)

                # Add labels for top 5 clusters in each subplot
                if add_labels and topic_words and cluster_centers is not None:
                    unique, counts = np.unique(cluster_labels, return_counts=True)
                    cluster_sizes = dict(zip(unique, counts))
                    top_clusters = sorted(cluster_sizes.keys(), key=lambda k: cluster_sizes[k], reverse=True)[:5]

                    for label in top_clusters:
                        if label in topic_words and label < len(cluster_centers):
                            words = topic_words[label]
                            # Use just first word for compact display
                            label_text = words[0].split()[0] if words else ""
                            if len(label_text) > 10:
                                label_text = label_text[:8] + '..'
                            center = cluster_centers[label]
                            ax.annotate(
                                label_text,
                                xy=(center[0], center[1]),
                                fontsize=6,
                                fontweight='bold',
                                ha='center', va='center',
                                color='black',
                                bbox=dict(
                                    boxstyle='round,pad=0.2',
                                    facecolor='white',
                                    edgecolor='gray',
                                    alpha=0.8,
                                    linewidth=0.5
                                )
                            )
            else:
                ax.scatter(coords[:, 0], coords[:, 1],
                          c=LAYER_COLORS[layer_num], s=25, alpha=0.8)

            ax.set_title(f'Layer {layer_num}\n({len(coords):,})', fontsize=11, fontweight='bold', pad=8)
            ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#cccccc')

    plt.suptitle('SAE Feature UMAP Projections - All 18 Layers',
                fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "umap_layer_grid_paper.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    return output_path


def plot_single_cluster_focus(layer_num: int = 12, dpi: int = 300):
    # Zoomed view of specific clusters for detail with concept labels
    data = load_layer_data(layer_num)
    if data is None:
        return None

    coords = data['coords']
    cluster_labels = data['cluster_labels_30']
    topic_words = data.get('topic_words_30')

    if cluster_labels is None:
        return None

    # Find the 5 largest clusters
    unique, counts = np.unique(cluster_labels, return_counts=True)
    top_clusters = unique[np.argsort(counts)[-5:]]

    fig, axes = plt.subplots(1, 5, figsize=(25, 6), dpi=dpi)

    for idx, cluster_id in enumerate(top_clusters):
        ax = axes[idx]
        mask = cluster_labels == cluster_id
        cluster_coords = coords[mask]

        # All points in gray
        ax.scatter(coords[:, 0], coords[:, 1], c='#e0e0e0', s=8, alpha=0.3)

        # Highlighted cluster
        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                  c=CLUSTER_COLORS[idx], s=60, alpha=0.9,
                  edgecolors='black', linewidth=0.8)

        # Zoom to cluster
        x_min, x_max = cluster_coords[:, 0].min(), cluster_coords[:, 0].max()
        y_min, y_max = cluster_coords[:, 1].min(), cluster_coords[:, 1].max()
        pad = 0.3 * max(x_max - x_min, y_max - y_min)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

        # Get concept label from topic_words
        concept_label = ""
        if topic_words and cluster_id in topic_words:
            words = topic_words[cluster_id][:3]  # Top 3 words
            concept_label = ', '.join(words)

        ax.set_title(f'Cluster {cluster_id}: {concept_label}\n({len(cluster_coords)} features)',
                    fontsize=11, fontweight='bold', wrap=True)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle(f'Layer {layer_num}: Largest Feature Clusters',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"umap_layer_{layer_num}_clusters_zoomed.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    return output_path


if __name__ == "__main__":
    print("Generating paper-quality UMAP figures...")
    print(f"Output: {OUTPUT_DIR}\n")

    # All layers combined
    print("1. All layers combined...")
    plot_all_layers_paper(dpi=300)

    # Grid view
    print("\n2. Layer grid...")
    plot_layer_grid_paper(dpi=300)

    # Cluster zoom for layer 12
    print("\n3. Cluster zoom (Layer 12)...")
    plot_single_cluster_focus(12, dpi=300)

    # Individual layers with large points
    print("\n4. Individual layers (large points)...")
    for layer in range(18):
        plot_layer_large(layer, dpi=300, point_size=40)

    print(f"\n✨ Done! Figures in: {OUTPUT_DIR}")
