"""
plot_heterarchy.py

Generates Figure 1: Conceptual illustration of the heterarchical organization of thoughtseeds.
"""

import random
import logging
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import os
import io
from PIL import Image
from pathlib import Path
import plotting_utils as pu

def add_weights_and_nodes(G, domain_config, seed=None):
    if seed is not None:
        random.seed(seed)

    for domain_id, config in enumerate(domain_config):
        domain_color = config['domain_color']
        num_ensembles = config['num_ensembles']
        num_NPs_range = config['num_NPs']

        G.add_node(f'NPD{domain_id}', label='NPD', size=700, color=domain_color, hollow=True, thickness=2)

        # Create Superordinate Ensembles with significance and randomly assign packets to each
        for ensemble_id in range(num_ensembles):
            ensemble_significance = random.choice([0.5, 0.6, 0.7, 0.8])
            total_complexity = 0

            num_NPs = random.randint(num_NPs_range[0], num_NPs_range[1])
            np_nodes = []
            for packet_id in range(num_NPs):
                packet_complexity = random.randint(1, 10)
                total_complexity += packet_complexity
                np_node = f'NP{domain_id}_{ensemble_id}_{packet_id}'
                np_nodes.append(np_node)
                
                # Assign color of SE to NP
                np_color = domain_color
                
                G.add_node(np_node, label='NP', size=packet_complexity * 100, color=np_color, hollow=False)
                weight = random.choice([0.3, 0.4, 0.5])
                G.add_edge(f'Se{domain_id}_{ensemble_id}', np_node, weight=weight)

            ensemble_size = total_complexity * 100
            G.add_node(f'Se{domain_id}_{ensemble_id}', label='Se', size=ensemble_size, color=domain_color, hollow=True, thickness=1)
            G.add_edge(f'NPD{domain_id}', f'Se{domain_id}_{ensemble_id}', weight=ensemble_significance)

            # Add random connections between NPs within the same SE
            for i in range(len(np_nodes)):
                for j in range(i + 1, len(np_nodes)):
                    if random.random() > 0.5:  # Adjust probability as needed
                        G.add_edge(np_nodes[i], np_nodes[j], weight=random.choice([0.2, 0.3, 0.4]))

        # Add connections between SEs within the same NPD
        for i in range(num_ensembles):
            for j in range(i + 1, num_ensembles):
                if random.random() > 0.2:
                    G.add_edge(f'Se{domain_id}_{i}', f'Se{domain_id}_{j}', weight=random.choice([0.4, 0.5, 0.6]))

    # Add weaker connections between different NPDs
    for i in range(len(domain_config)):
        for j in range(i + 1, len(domain_config)):
            G.add_edge(f'NPD{i}', f'NPD{j}', weight=random.choice([0.2, 0.3, 0.4]))
            
def ensure_np_connections(G, min_weight=1):
    se_nodes = [node for node in G.nodes if G.nodes[node]['label'] == 'Se']
    np_nodes = [node for node in G.nodes if G.nodes[node]['label'] == 'NP']

    # Ensure all NPs are connected to at least one SE
    for np_node in np_nodes:
        se_neighbors = [n for n in G.neighbors(np_node) if G.nodes[n]['label'] == 'Se']
        if not se_neighbors:
            random_se = random.choice(se_nodes)
            G.add_edge(np_node, random_se, weight=min_weight)
            # print(f"Connected NP {np_node} to SE {random_se}")

    # Add connections between NPs within the same SE
    for se_node in se_nodes:
        np_neighbors = [n for n in G.neighbors(se_node) if G.nodes[n]['label'] == 'NP']
        if len(np_neighbors) > 1:
            for i in range(len(np_neighbors)):
                for j in range(i + 1, len(np_neighbors)):
                    if not G.has_edge(np_neighbors[i], np_neighbors[j]):
                        G.add_edge(np_neighbors[i], np_neighbors[j], weight=min_weight)
                        # print(f"Connected NP {np_neighbors[i]} to NP {np_neighbors[j]} within SE {se_node}")

def plot_graph(G, show_NPD_labels=True, show_SE_labels=False, show_NP_labels=True):
    pu.set_plot_style()
    ensure_np_connections(G, min_weight=0.1)

    pos = nx.kamada_kawai_layout(G)  # Adjust parameters for better separation

    labels = {}
    for node in G.nodes:
        if show_NPD_labels and G.nodes[node]['label'] == 'NPD':
            labels[node] = node
        elif show_SE_labels and G.nodes[node]['label'] == 'Se':
            labels[node] = node
        elif show_NP_labels and G.nodes[node]['label'] == 'NP':
            labels[node] = node

    # Create figure
    plt.figure(figsize=(12, 8))
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')

    # Define alpha values
    alpha_values = [0.3, 0.5, 0.65, 0.8]

    # Draw envelopes around NPs belonging to each SE
    for i, node in enumerate(G.nodes):
        if G.nodes[node]['label'] == 'Se':
            np_nodes = [n for n in G.neighbors(node) if G.nodes[n]['label'] == 'NP']
            if len(np_nodes) > 2:  # Convex hull requires at least 3 points
                np_positions = np.array([pos[n] for n in np_nodes])
                hull = ConvexHull(np_positions)
                for simplex in hull.simplices:
                    plt.plot(np_positions[simplex, 0], np_positions[simplex, 1], color=G.nodes[node]['color'], alpha=0.3, linewidth=2)
                # Project the Convex Hull to the SE point
                for vertex in hull.vertices:
                    plt.plot([np_positions[vertex, 0], pos[node][0]], [np_positions[vertex, 1], pos[node][1]], color=G.nodes[node]['color'], alpha=0.3, linewidth=2)
            
            # Assign alpha values and random sizes to NPs within each SE
            alpha = alpha_values[i % len(alpha_values)]
            for np_node in np_nodes:
                node_size = random.uniform(0.5, 1.5) * 100  # Scale size for better visualization
                nx.draw_networkx_nodes(G, pos, nodelist=[np_node], node_color=G.nodes[node]['color'], alpha=alpha, node_size=node_size)

            # Draw small square at SE position
            plt.scatter(pos[node][0], pos[node][1], edgecolors=G.nodes[node]['color'], facecolors='none', s=100, marker='s')
    
    # Draw envelopes around SEs belonging to each NPD
    for node in G.nodes:
        if G.nodes[node]['label'] == 'NPD':
            se_nodes = [n for n in G.neighbors(node) if G.nodes[n]['label'] == 'Se']
            if len(se_nodes) > 2:  # Convex hull requires at least 3 points
                se_positions = np.array([pos[n] for n in se_nodes])
                hull = ConvexHull(se_positions)
                for simplex in hull.simplices:
                    plt.plot(se_positions[simplex, 0], se_positions[simplex, 1], color=G.nodes[node]['color'], alpha=0.2, linestyle='dotted')
                # Project the Convex Hull to the NPD point
                for vertex in hull.vertices:
                    plt.plot([se_positions[vertex, 0], pos[node][0]], [se_positions[vertex, 1], pos[node][1]], color=G.nodes[node]['color'], alpha=0.5, linewidth=3, linestyle='--')

            plt.scatter(pos[node][0], pos[node][1], color=G.nodes[node]['color'], s=100, marker='s')
            
    # Join the top-level KD_SE nodes
    kd_se_nodes = [node for node in G.nodes if G.nodes[node].get('label') == 'NPD']
    for i in range(len(kd_se_nodes)):
        for j in range(i + 1, len(kd_se_nodes)):
            plt.plot([pos[kd_se_nodes[i]][0], pos[kd_se_nodes[j]][0]], [pos[kd_se_nodes[i]][1], pos[kd_se_nodes[j]][1]], color='grey', linestyle='dotted')

    # Finalize and show the plot
    plt.title("Superordinate Ensembles at Nested Scales of Heterarchy", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='grey', markerfacecolor='grey', alpha=0.7, markersize=14, label='Non-local Superordinate Ensemble Heterachy'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='grey', markersize=14, label='Local Superordinate Ensemble of NPs'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='grey', alpha=0.7, markersize=14, label='Neuronal Packet(NP)'),
    ]

    plt.legend(handles=legend_elements, loc='upper right', prop={'weight': 'bold'})
    
    # Save the plot
    out_path = Path(pu.PLOT_DIR) / 'Fig1_Heterarchy.jpeg'
    
    # Save logic with quality control
    max_bytes = 1_000_000
    quality = 90
    min_quality = 30
    while True:
        try:
            # Save to an in-memory PNG first, then convert to JPEG with Pillow
            buf = io.BytesIO()
            plt.savefig(buf, dpi=300, bbox_inches='tight', pad_inches=0.08, format='png')
            buf.seek(0)
            with Image.open(buf) as im:
                rgb = im.convert('RGB')
                rgb.save(out_path, format='JPEG', quality=quality, optimize=True)
        except Exception:
            # Fallback: let Matplotlib write directly (no quality control)
            try:
                plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.08)
            except Exception:
                break

        try:
            size = os.path.getsize(out_path)
        except OSError:
            break
        if size <= max_bytes or quality <= min_quality:
            break
        quality = max(min_quality, int(quality * 0.8))

    logging.info("Saved heterarchy plot to %s", out_path)

if __name__ == "__main__":
    domain_config = [
        {'domain_color': 'blue', 'num_ensembles': 4, 'num_NPs': (5, 8)},  # Increased range for blue domain
        {'domain_color': 'green', 'num_ensembles': 5, 'num_NPs': (4, 7)},
        {'domain_color': 'red', 'num_ensembles': 4, 'num_NPs': (3, 8)},
        {'domain_color': 'purple', 'num_ensembles': 4, 'num_NPs': (3, 6)},
    ]

    G = nx.Graph()
    add_weights_and_nodes(G, domain_config, seed=42)
    plot_graph(G, show_NPD_labels=False, show_SE_labels=False, show_NP_labels=False)
