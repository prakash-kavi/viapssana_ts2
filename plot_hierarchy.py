"""
plot_hierarchy.py

Generates Figure 5: Hierarchical Dynamics Snapshots (Tail Window).
Includes:
- (A) Novice Profile
- (B) Expert Profile

Each panel visualizes:
- Meta-awareness (Level 3)
- Dominant Thoughtseeds (Level 2)
- Network Activations (Level 1)
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from plotting_utils import (
    set_plot_style, load_json_data, get_tail_stats, smooth_series,
    PLOT_DIR, TAIL_STEPS, STATE_COLORS, NETWORK_COLORS, 
    STATE_SHORT_NAMES, THOUGHTSEED_COLORS, STATE_DISPLAY_NAMES
)

NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']
THOUGHTSEEDS = ['breath_focus', 'pending_tasks', 'pain_discomfort', 'self_reflection', 'equanimity']

def plot_hierarchy(stats_data, save_path=None, title=None):
    """
    Hierarchical plot (meta-awareness, dominant thoughtseed, networks).
    Expects stats_data to be the transition/summary stats dict.
    """
    set_plot_style()
    required_fields = ['state_history', 'meta_awareness_history', 'network_activations_history', 'dominant_ts_history']
    for field in required_fields:
        if field not in stats_data:
            logging.error("Required data '%s' missing for hierarchy plot", field)
            return

    total_len = len(stats_data['state_history'])
    start_idx = max(0, total_len - TAIL_STEPS)

    state_history = stats_data['state_history'][start_idx:]
    meta_awareness_raw = stats_data['meta_awareness_history'][start_idx:]
    meta_awareness = smooth_series(meta_awareness_raw, alpha=0.6)
    dom_ts_hist = stats_data['dominant_ts_history'][start_idx:]
    net_hist = stats_data['network_activations_history'][start_idx:]
    hazard_full = stats_data.get('van_hazard_history')
    hazard = hazard_full[start_idx:] if hazard_full and len(hazard_full) == total_len else None

    time_steps = np.arange(len(state_history))

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.5], figure=fig)

    # --- Level 3: Metacognition ---
    ax1 = fig.add_subplot(gs[0])

    # Draw hazard background first (if present)
    if hazard is not None and len(hazard) == len(time_steps):
        hazard = np.array(hazard)
        hmin, hmax = np.min(hazard), np.max(hazard) + 1e-9
        hnorm = (hazard - hmin) / (hmax - hmin)
        ax1.fill_between(time_steps, 0, hnorm, color='#B77FB4', alpha=0.12, label='Hazard', zorder=0)

    # Shade area under the meta-awareness curve and draw the line on top
    ax1.fill_between(time_steps, 0, meta_awareness, color='#4363d8', alpha=0.12, zorder=1)
    ax1.plot(time_steps, meta_awareness, color='#4363d8', linewidth=2, label='Meta', zorder=2)

    ax1.set_ylabel('Meta-Awareness', fontsize=12)
    ax1.set_title('Level 3: Metacognition', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # --- Level 2: Dominant Thoughtseed ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    thoughtseeds = THOUGHTSEEDS
    ts_mapping = {ts: i for i, ts in enumerate(thoughtseeds)}
    for i, ts in enumerate(dom_ts_hist):
        ax2.scatter(i, ts_mapping.get(ts, 0), color=THOUGHTSEED_COLORS.get(ts, '#333333'), s=25,
                   edgecolors='white', linewidth=0.5, alpha=0.8)
    
    ax2.set_yticks(range(len(THOUGHTSEEDS)))
    ax2.set_yticklabels(THOUGHTSEEDS)
    ax2.invert_yaxis()
    ax2.set_ylabel('Dominant Thoughtseed', fontsize=12)
    ax2.set_title('Level 2: Dominant Thoughtseed', fontsize=14, fontweight='bold', pad=-15)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    # --- Level 1: Network Dynamics ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for net in NETWORKS:
        net_acts = [n.get(net, 0.0) for n in net_hist]
        if len(net_acts) == 0:
            continue
        smoothed = np.zeros_like(net_acts, dtype=float)
        alpha = 0.3
        smoothed[0] = net_acts[0]
        for j in range(1, len(net_acts)):
            smoothed[j] = (1 - alpha) * smoothed[j-1] + alpha * net_acts[j]
        ax3.plot(time_steps, smoothed, label=net, color=NETWORK_COLORS[net], linewidth=2)

    # --- State Transitions (Vertical Lines & Labels) ---
    prev_state = None
    for i, state in enumerate(state_history):
        if state != prev_state:
            state_label = STATE_SHORT_NAMES.get(state, state)
            ax1.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            ax2.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            ax3.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            
            # Label on top plot
            ax1.text(i, -0.05, state_label,
                     rotation=90, fontsize=9, color=STATE_COLORS.get(state, '#000000'),
                     transform=ax1.get_xaxis_transform(), ha='center', va='top')
            prev_state = state

    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('Network Activation', fontsize=12)
    ax3.set_title('Level 1: Network Dynamics', fontsize=14, fontweight='bold', pad=-25)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, bottom=0.12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info("Saved hierarchy plot to %s", save_path)
    plt.close()

if __name__ == "__main__":
    # Load Data
    nov_ts, nov_ai, nov_stats = load_json_data("novice")
    exp_ts, exp_ai, exp_stats = load_json_data("expert")
    
    # Slice to Tail
    nov_tail = get_tail_stats(nov_stats)
    exp_tail = get_tail_stats(exp_stats)
    
    # Generate Panels
    plot_hierarchy(nov_tail, save_path=os.path.join(PLOT_DIR, "Fig4A_Hierarchy_Novice.png"), title="Novice Profile")
    plot_hierarchy(exp_tail, save_path=os.path.join(PLOT_DIR, "Fig4B_Hierarchy_Expert.png"), title="Expert Profile")
