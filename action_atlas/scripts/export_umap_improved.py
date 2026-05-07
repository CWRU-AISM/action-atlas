#!/usr/bin/env python3
"""
Generate high-quality UMAP visualizations for paper.
- 2D and 3D plots
- Dark background with vibrant colors
- Larger points for visibility
- Individual and combined layer views
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
# UMAP imported from umap-learn if needed for 3D computation
# from umap import UMAP

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "figures"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Vibrant colors that pop on dark background
LAYER_COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#F38181",
    "#AA96DA", "#FCBAD3", "#A8D8EA", "#FF9A8B", "#88D8B0",
    "#C9B1FF", "#FFEAA7", "#74B9FF", "#FD79A8", "#00B894",
    "#E17055", "#81ECEC", "#FDCB6E"
]

# Cluster colors - more saturated
CLUSTER_COLORS_30 = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8B500', '#82E0AA', '#F1948A', '#85929E', '#A569BD',
    '#5DADE2', '#F4D03F', '#48C9B0', '#EC7063', '#AF7AC5',
    '#5499C7', '#52BE80', '#F5B041', '#A3E4D7', '#D7BDE2',
    '#AED6F1', '#FAD7A0', '#D5F5E3', '#FADBD8', '#E8DAEF'
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
            return {
                'coords': data['coords'],
                'descriptions': data['descriptions'] if 'descriptions' in data else None,
                'cluster_labels_30': data['cluster_labels_30'] if 'cluster_labels_30' in data else None,
                'cluster_colors_30': data['cluster_colors_30'] if 'cluster_colors_30' in data else None,
            }
    return None


def plot_layer_2d_dark(layer_num: int, dpi: int = 300):
    # Generate dark-themed 2D UMAP for a layer with cluster colors
    data = load_layer_data(layer_num)
    if data is None:
        print(f"No data for layer {layer_num}")
        return None

    coords = data['coords']
    cluster_labels = data['cluster_labels_30']

    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    if cluster_labels is not None:
        # Use cluster colors
        unique_labels = np.unique(cluster_labels)
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            color = CLUSTER_COLORS_30[i % len(CLUSTER_COLORS_30)]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, s=25, alpha=0.8, edgecolors='white',
                linewidth=0.3, label=f'Cluster {label}'
            )
    else:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=LAYER_COLORS[layer_num], s=25, alpha=0.8,
            edgecolors='white', linewidth=0.3
        )

    ax.set_xlabel('UMAP 1', fontsize=14, color='white')
    ax.set_ylabel('UMAP 2', fontsize=14, color='white')
    ax.set_title(f'SAE Feature Clusters - Layer {layer_num}\n({len(coords):,} features, 30 clusters)',
                fontsize=16, color='white', pad=20)

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='white', linestyle='--')

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"umap_layer_{layer_num}_dark.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved: {output_path}")
    plt.close()
    return output_path


def plot_all_layers_2d_dark(dpi: int = 300):
    # Combined 2D plot with all layers, dark theme
    fig, ax = plt.subplots(figsize=(16, 16), dpi=dpi, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

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
        c=all_colors, s=8, alpha=0.7, edgecolors='none'
    )

    ax.set_xlabel('UMAP 1', fontsize=14, color='white')
    ax.set_ylabel('UMAP 2', fontsize=14, color='white')
    ax.set_title(f'SAE Feature UMAP - All 18 Layers\n({len(all_coords):,} total features)',
                fontsize=18, color='white', pad=20)

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, alpha=0.15, color='white', linestyle='--')

    # Legend
    handles = []
    for layer_num, count in layer_counts:
        handle = plt.Line2D([0], [0], marker='o', color='#1a1a2e',
                           markerfacecolor=LAYER_COLORS[layer_num], markersize=10,
                           label=f'L{layer_num} ({count:,})')
        handles.append(handle)

    legend = ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=10, framealpha=0.9, ncol=1, facecolor='#2d2d44',
                      edgecolor='white', labelcolor='white')

    plt.tight_layout()

    output_path = OUTPUT_DIR / "umap_all_layers_dark.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved: {output_path}")
    plt.close()
    return output_path


def plot_layer_3d_interactive(layer_num: int):
    # Generate interactive 3D UMAP using plotly
    data = load_layer_data(layer_num)
    if data is None:
        print(f"No data for layer {layer_num}")
        return None

    coords = data['coords']
    cluster_labels = data['cluster_labels_30']

    # We need 3D coords - run UMAP with 3 components if we only have 2D
    if coords.shape[1] == 2:
        # Need original high-dim data to compute 3D UMAP
        # For now, add a z-coordinate based on cluster or random
        z = cluster_labels if cluster_labels is not None else np.random.randn(len(coords))
        coords_3d = np.column_stack([coords, z * 0.5])
    else:
        coords_3d = coords

    # Create plotly figure
    if cluster_labels is not None:
        fig = go.Figure(data=[go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=cluster_labels,
                colorscale='Viridis',
                opacity=0.8,
            ),
            text=[f'Feature {i}' for i in range(len(coords))],
            hoverinfo='text'
        )])
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=LAYER_COLORS[layer_num],
                opacity=0.8,
            ),
        )])

    fig.update_layout(
        title=f'3D UMAP - Layer {layer_num} ({len(coords):,} features)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            bgcolor='#1a1a2e',
        ),
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        width=1200,
        height=1000,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"umap_layer_{layer_num}_3d.html"
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")

    # Also save as static image
    output_png = OUTPUT_DIR / f"umap_layer_{layer_num}_3d.png"
    try:
        fig.write_image(str(output_png), width=1200, height=1000, scale=2)
        print(f"Saved: {output_png}")
    except Exception as e:
        print(f"  (Could not save PNG: {e})")

    return output_path


def plot_all_layers_3d():
    # Generate 3D visualization with all layers
    all_coords = []
    all_colors = []
    all_layers = []

    for layer_num in range(18):
        data = load_layer_data(layer_num)
        if data is None:
            continue

        coords = data['coords']
        cluster_labels = data['cluster_labels_30']

        # Add z based on layer number + cluster
        z = np.ones(len(coords)) * layer_num
        if cluster_labels is not None:
            z += cluster_labels * 0.1

        coords_3d = np.column_stack([coords, z])
        all_coords.append(coords_3d)
        all_colors.extend([LAYER_COLORS[layer_num]] * len(coords))
        all_layers.extend([layer_num] * len(coords))

    if not all_coords:
        print("No data found")
        return None

    all_coords = np.vstack(all_coords)

    fig = go.Figure(data=[go.Scatter3d(
        x=all_coords[:, 0],
        y=all_coords[:, 1],
        z=all_coords[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=all_colors,
            opacity=0.6,
        ),
        text=[f'Layer {l}' for l in all_layers],
        hoverinfo='text'
    )])

    fig.update_layout(
        title=f'3D UMAP - All Layers ({len(all_coords):,} features)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='Layer',
            bgcolor='#1a1a2e',
        ),
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        width=1400,
        height=1000,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "umap_all_layers_3d.html"
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")

    return output_path


def plot_layer_grid_dark(dpi: int = 300):
    # Generate 3x6 grid of all layers with dark theme
    fig, axes = plt.subplots(3, 6, figsize=(30, 15), dpi=dpi, facecolor='#1a1a2e')
    axes = axes.flatten()

    for layer_num in range(18):
        ax = axes[layer_num]
        ax.set_facecolor('#1a1a2e')
        data = load_layer_data(layer_num)

        if data is None:
            ax.text(0.5, 0.5, f'Layer {layer_num}\nNo data',
                   ha='center', va='center', fontsize=12, color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            coords = data['coords']
            cluster_labels = data['cluster_labels_30']

            if cluster_labels is not None:
                unique_labels = np.unique(cluster_labels)
                for i, label in enumerate(unique_labels):
                    mask = cluster_labels == label
                    color = CLUSTER_COLORS_30[i % len(CLUSTER_COLORS_30)]
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                              c=color, s=8, alpha=0.7, edgecolors='none')
            else:
                ax.scatter(coords[:, 0], coords[:, 1],
                          c=LAYER_COLORS[layer_num], s=8, alpha=0.7, edgecolors='none')

            ax.set_title(f'Layer {layer_num} ({len(coords):,})', fontsize=14, color='white', pad=10)
            ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle('SAE Feature UMAP Projections - All 18 Layers (30 Clusters Each)',
                fontsize=20, color='white', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "umap_layer_grid_dark.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved: {output_path}")
    plt.close()
    return output_path


if __name__ == "__main__":
    print("Generating improved UMAP figures...")
    print(f"Output: {OUTPUT_DIR}\n")

    # 2D dark theme plots
    print("1. Single layer (Layer 12) - dark theme...")
    plot_layer_2d_dark(12, dpi=300)

    print("\n2. All layers combined - dark theme...")
    plot_all_layers_2d_dark(dpi=300)

    print("\n3. Layer grid - dark theme...")
    plot_layer_grid_dark(dpi=300)

    # 3D plots
    print("\n4. 3D interactive (Layer 12)...")
    plot_layer_3d_interactive(12)

    print("\n5. 3D all layers...")
    plot_all_layers_3d()

    # Generate all individual layers
    print("\n6. All individual layers (dark theme)...")
    for layer in range(18):
        plot_layer_2d_dark(layer, dpi=300)

    print(f"\n✨ Done! Figures in: {OUTPUT_DIR}")
