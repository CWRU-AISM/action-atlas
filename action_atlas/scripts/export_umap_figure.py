#!/usr/bin/env python3
"""
Export high-resolution UMAP scatter plot for paper figures.
Generates publication-quality images at 300+ DPI.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path
import json

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "figures"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Layer colors for multi-layer visualization
LAYER_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1"
]

def load_layer_data(layer_num: int):
    """Load UMAP data for a specific layer."""
    possible_files = [
        DATA_DIR / f"hierarchical_clustering_layer_{layer_num}_goal.npz",
        DATA_DIR / f"hierarchical_clustering_layer_{layer_num}_concepts.npz",
    ]

    for npz_path in possible_files:
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            return {
                'coords': data['coords'],
                'descriptions': data['descriptions'],
                'cluster_labels_10': data.get('cluster_labels_10'),
                'cluster_labels_30': data.get('cluster_labels_30'),
                'cluster_colors_10': data.get('cluster_colors_10'),
                'cluster_colors_30': data.get('cluster_colors_30'),
            }
    return None


def plot_single_layer(layer_num: int, cluster_level: int = 10, dpi: int = 300):
    """Generate UMAP scatter plot for a single layer."""
    data = load_layer_data(layer_num)
    if data is None:
        print(f"No data found for layer {layer_num}")
        return None

    coords = data['coords']

    # Use cluster colors if available
    if f'cluster_colors_{cluster_level}' in data and data[f'cluster_colors_{cluster_level}'] is not None:
        colors = data[f'cluster_colors_{cluster_level}']
    else:
        colors = LAYER_COLORS[layer_num % len(LAYER_COLORS)]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors if isinstance(colors, np.ndarray) else None,
        color=colors if isinstance(colors, str) else None,
        s=8, alpha=0.6, edgecolors='none'
    )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'SAE Feature Clusters - Layer {layer_num}\n({len(coords):,} features)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Remove axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"umap_layer_{layer_num}_clusters_{cluster_level}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close()
    return output_path


def plot_all_layers(dpi: int = 300):
    """Generate combined UMAP visualization with all layers."""
    fig, ax = plt.subplots(figsize=(14, 14), dpi=dpi)

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
        print("No data found for any layer")
        return None

    all_coords = np.vstack(all_coords)

    scatter = ax.scatter(
        all_coords[:, 0], all_coords[:, 1],
        c=all_colors, s=4, alpha=0.5, edgecolors='none'
    )

    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.set_title(f'SAE Feature Clusters - All Layers\n({len(all_coords):,} total features)', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    handles = []
    for layer_num, count in layer_counts:
        color = LAYER_COLORS[layer_num]
        handle = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color, markersize=8,
                           label=f'L{layer_num} ({count:,})')
        handles.append(handle)

    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
             fontsize=9, framealpha=0.9, ncol=1)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "umap_all_layers.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close()
    return output_path


def plot_layer_grid(dpi: int = 300):
    """Generate a grid of UMAP plots for all layers."""
    fig, axes = plt.subplots(3, 6, figsize=(24, 12), dpi=dpi)
    axes = axes.flatten()

    for layer_num in range(18):
        ax = axes[layer_num]
        data = load_layer_data(layer_num)

        if data is None:
            ax.text(0.5, 0.5, f'Layer {layer_num}\nNo data',
                   ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            coords = data['coords']
            color = LAYER_COLORS[layer_num]

            ax.scatter(coords[:, 0], coords[:, 1],
                      c=color, s=2, alpha=0.5, edgecolors='none')
            ax.set_title(f'Layer {layer_num} ({len(coords):,})', fontsize=10)
            ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle('SAE Feature UMAP Projections Across All Layers', fontsize=16, y=1.02)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "umap_layer_grid.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close()
    return output_path


if __name__ == "__main__":
    print("Generating UMAP figures for paper...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate single layer plot (layer 12 as example)
    print("1. Single layer plot (Layer 12)...")
    plot_single_layer(12, cluster_level=10, dpi=300)

    # Generate all layers combined
    print("\n2. All layers combined plot...")
    plot_all_layers(dpi=300)

    # Generate layer grid
    print("\n3. Layer grid plot...")
    plot_layer_grid(dpi=300)

    print(f"\n✨ Done! Figures saved to: {OUTPUT_DIR}")
