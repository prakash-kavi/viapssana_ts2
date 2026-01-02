"""
plot_diagnostics.py

Generates Figure 4: Reference Profile Diagnostics (Tail Window).
Includes:
- (A) Time Series (Network Activations & Free Energy)
- (B) Free Energy Bar Chart
- (C) Network Radar Plots
- (D) Dwell Time Bar Chart
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from plotting_utils import (
    set_plot_style, load_json_data, get_tail_stats, smooth_series,
    PLOT_DIR, TAIL_STEPS, STATE_COLORS, NETWORK_COLORS, 
    STATE_SHORT_NAMES, STATE_DISPLAY_NAMES
)

NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']
STATES = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]

def plot_time_series(novice_stats, expert_stats, save_path=None):
    """Panel A: Tail-window Dynamics"""
    set_plot_style()
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])

    axes = []

    for col, (level, stats) in enumerate([('Novice', novice_stats), ('Expert', expert_stats)]):
        # Data is already sliced to tail by get_tail_stats
        state_history = stats['state_history']
        net_hist = stats['network_activations_history']
        fe_raw = np.array(stats.get('free_energy_history', []))
        time_steps = np.arange(len(state_history))

        # 1. Network Activations
        ax1 = fig.add_subplot(gs[0, col])
        for net in NETWORKS:
            net_acts = [n.get(net, 0.0) for n in net_hist]
            ax1.plot(time_steps, net_acts, label=f"{net}", color=NETWORK_COLORS[net], linewidth=1.5)

        # State markers
        prev_state = None
        for i, state in enumerate(state_history):
            if state != prev_state:
                state_label = STATE_SHORT_NAMES.get(state, state)
                ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
                if i < len(state_history) - 5: # Avoid label at very end
                    ax1.text(i, 1.05, state_label, rotation=90, fontsize=9, 
                             color=STATE_COLORS.get(state, '#000000'),
                             transform=ax1.get_xaxis_transform(), ha='center')
                prev_state = state

        ax1.set_ylim(0, 1.15)
        ax1.set_title(f"Network Activations ({level})", fontsize=14, fontweight='bold')
        ax1.set_ylabel('Activation', fontsize=12)
        if col == 1: ax1.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Free Energy
        ax2 = fig.add_subplot(gs[1, col], sharex=ax1)
        fe_smooth = smooth_series(fe_raw)
        
        ax2.plot(time_steps, fe_smooth, color='#E74C3C', label="Free Energy (smoothed)", linewidth=2)
        if len(fe_raw) > 0:
            ax2.plot(time_steps, fe_raw, color='#E74C3C', alpha=0.25, linewidth=1)

        # State background shading
        prev_state = state_history[0] if len(state_history) > 0 else None
        start_idx = 0
        for i, state in enumerate(state_history):
            if state != prev_state or i == len(state_history)-1:
                ax2.axvspan(start_idx, i, alpha=0.1, color=STATE_COLORS.get(prev_state, '#cccccc'))
                start_idx = i
                prev_state = state

        ax2.set_title(f"Free Energy ({level})", fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timestep (Tail Window)', fontsize=12)
        ax2.set_ylabel('Free Energy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        axes.append(ax2)

    # Unify Y-axis for Free Energy
    y_mins = [ax.get_ylim()[0] for ax in axes]
    y_maxs = [ax.get_ylim()[1] for ax in axes]
    for ax in axes:
        ax.set_ylim(min(y_mins), max(y_maxs))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info("Saved Time Series to %s", save_path)
    plt.close()

def plot_free_energy_bar(novice_stats, expert_stats, save_path=None):
    """Panel B: Free Energy distribution per state as boxplots (novice vs expert)."""
    set_plot_style()

    def _collect_fe_by_state(stats):
        states = stats.get("state_history", [])
        fe = stats.get("free_energy_history", [])
        by_state = {s: [] for s in STATES}
        for s, f in zip(states, fe):
            if s in by_state:
                try:
                    by_state[s].append(float(f))
                except Exception:
                    pass
        return by_state

    nov_by_state = _collect_fe_by_state(novice_stats)
    exp_by_state = _collect_fe_by_state(expert_stats)

    # Prepare data for bar chart style similar to Fig4D (means + std)
    x = np.arange(len(STATES))
    width = 0.35

    nov_vals = [np.mean(nov_by_state[s]) if nov_by_state[s] else 0.0 for s in STATES]
    exp_vals = [np.mean(exp_by_state[s]) if exp_by_state[s] else 0.0 for s in STATES]
    nov_err = [np.std(nov_by_state[s]) if nov_by_state[s] else 0.0 for s in STATES]
    exp_err = [np.std(exp_by_state[s]) if exp_by_state[s] else 0.0 for s in STATES]

    fig, ax = plt.subplots(figsize=(10, 6))
    nov_bars = ax.bar(x - width/2, nov_vals, width, yerr=nov_err, capsize=5,
                      label='Novice', color=[STATE_COLORS[s] for s in STATES], alpha=0.7,
                      edgecolor='black', linewidth=1)
    exp_bars = ax.bar(x + width/2, exp_vals, width, yerr=exp_err, capsize=5,
                      label='Expert', color=[STATE_COLORS[s] for s in STATES], alpha=0.4,
                      hatch='//', edgecolor='black', linewidth=1)

    ax.set_ylabel('Free Energy', fontsize=12, fontweight='bold')
    ax.set_title('Free Energy: Mean and Variability Across States', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=11)

    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info("Saved FE Bar to %s", save_path)
    plt.close()

def plot_network_radar(novice_ts, expert_ts, save_path=None):
    """Panel C: Network Radar Plots"""
    set_plot_style()
    def _extract_net(ts_data):
        # Try to find network expectations
        if "network_expectations" in ts_data: # Sometimes in active_inference_params
             return ts_data["network_expectations"]
        # Or in thoughtseed params if structured that way
        return ts_data.get("network_expectations", {}) # Fallback

    # Note: In this codebase, network expectations seem to be in active_inference_params (ai_data)
    # But the original code passed 'novice_ts'. Let's handle both.
    nov_nets = _extract_net(novice_ts)
    exp_nets = _extract_net(expert_ts)

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Network Activation Profiles', fontsize=18, fontweight='bold')

    angles = np.linspace(0, 2*np.pi, len(NETWORKS), endpoint=False).tolist()
    angles += angles[:1]

    for i, state in enumerate(STATES):
        ax = fig.add_subplot(2, 2, i+1, polar=True)

        nov_state = nov_nets.get(state, {})
        exp_state = exp_nets.get(state, {})

        nov_vals = [float(nov_state.get(net, 0.0)) for net in NETWORKS]
        exp_vals = [float(exp_state.get(net, 0.0)) for net in NETWORKS]
        nov_vals += nov_vals[:1]
        exp_vals += exp_vals[:1]

        # State-coloured lines/fills (Novice dashed, Expert solid) — restored previous scheme
        ax.plot(angles, nov_vals, color=STATE_COLORS[state], linewidth=2.6, linestyle='--', label="Novice")
        ax.fill(angles, nov_vals, color=STATE_COLORS[state], alpha=0.22)
        ax.plot(angles, exp_vals, color=STATE_COLORS[state], linewidth=2.8, linestyle='-', label="Expert")
        ax.fill(angles, exp_vals, color=STATE_COLORS[state], alpha=0.16)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(NETWORKS, fontsize=13, fontweight='bold')
        # Improve radial label size (y-axis labels)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(STATE_DISPLAY_NAMES[state], fontsize=15, fontweight='bold', pad=18)
        ax.grid(True, linestyle='--', alpha=0.7)
        # Slightly increase label padding for polar plots
        for lbl in ax.get_xticklabels():
            lbl.set_y(0.02)

    labels = ["Expert", "Novice"]
    handles = [
        plt.Line2D([0], [0], color='black', linewidth=2.6, label=labels[0]),
        plt.Line2D([0], [0], color='black', linewidth=2.2, linestyle='--', label=labels[1])
    ]
    fig.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.08), ncol=2, fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info("Saved Radar to %s", save_path)
    plt.close()

def get_dwell_times(stats):
    state_history = stats.get("state_history", [])
    if not state_history: return {s: [] for s in STATES}
    
    dwells = {s: [] for s in STATES}
    if not state_history: return dwells
    
    current_state = state_history[0]
    count = 0
    for s in state_history:
        if s == current_state:
            count += 1
        else:
            dwells[current_state].append(count)
            current_state = s
            count = 1
    dwells[current_state].append(count)
    return dwells

def plot_dwell_times(novice_stats, expert_stats, save_path=None):
    """Panel D: Dwell Time Bar Chart"""
    set_plot_style()
    
    nov_dwells = get_dwell_times(novice_stats)
    exp_dwells = get_dwell_times(expert_stats)

    nov_means = [np.mean(nov_dwells[s]) if nov_dwells[s] else 0 for s in STATES]
    exp_means = [np.mean(exp_dwells[s]) if exp_dwells[s] else 0 for s in STATES]
    nov_err = [np.std(nov_dwells[s]) if nov_dwells[s] else 0 for s in STATES]
    exp_err = [np.std(exp_dwells[s]) if exp_dwells[s] else 0 for s in STATES]

    x = np.arange(len(STATES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    nov_bars = ax.bar(x - width/2, nov_means, width, yerr=nov_err, capsize=5, label='Novice', 
                      color=[STATE_COLORS[s] for s in STATES], alpha=0.7, edgecolor='black', linewidth=1)
    exp_bars = ax.bar(x + width/2, exp_means, width, yerr=exp_err, capsize=5, label='Expert', 
                      color=[STATE_COLORS[s] for s in STATES], alpha=0.4, hatch='//', edgecolor='black', linewidth=1)

    ax.set_ylabel('Average Dwell Time (Timesteps)', fontsize=12, fontweight='bold')
    ax.set_title('Average Dwell Time per State', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=11)
    ax.legend()

    # Simple bar plot: no numeric annotations on bars (publication style)

    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info("Saved Dwell Times to %s", save_path)
    plt.close()

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Load Data
    nov_ts, nov_ai, nov_stats = load_json_data("novice")
    exp_ts, exp_ai, exp_stats = load_json_data("expert")
    
    # Slice to Tail
    nov_tail = get_tail_stats(nov_stats)
    exp_tail = get_tail_stats(exp_stats)
    
    # Generate Panels
    plot_time_series(nov_tail, exp_tail, os.path.join(PLOT_DIR, "FigS1C_TimeSeries.png"))
    plot_free_energy_bar(nov_tail, exp_tail, os.path.join(PLOT_DIR, "Fig3A_FreeEnergy.png"))
    # Note: Radar needs network expectations, usually in ai_data for this model
    plot_network_radar(nov_ai, exp_ai, os.path.join(PLOT_DIR, "Fig3B_Radar.png"))
    plot_dwell_times(nov_tail, exp_tail, os.path.join(PLOT_DIR, "Fig3C_DwellTime.png"))
