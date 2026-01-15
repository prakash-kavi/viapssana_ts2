"""
verify_stats.py

Verifies and computes statistics for Free Energy, network activations,
dwell times, meta-awareness, and attractor dynamics for both novice and expert
profiles.

Usage:
    python verify_stats.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from viz import plotting_utils
from viz.plotting_utils import load_json_data, STATE_DISPLAY_NAMES

# State name mappings
STATE_MAP = {
    "attend_breath": "Breath Focus",
    "mind_wandering": "Mind Wandering",
    "meta_awareness": "Meta-Awareness",
    "redirect_breath": "Redirected Attention"
}

NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']


def calculate_stats(cohort):
    """
    Calculate and display comprehensive statistics for a given cohort.
    
    Args:
        cohort: Either 'novice' or 'expert'
    """
    print(f"--- Processing {cohort} ---")
    ts_data, ai_data, stats_data = load_json_data(cohort)
    
    # Validate data availability
    if 'state_history' not in stats_data:
        print(f"Error: No state_history found for {cohort}")
        return

    state_history = stats_data['state_history']
    fe_history = stats_data.get('free_energy_history', [])
    net_history = stats_data.get('network_activations_history', [])

    # Ensure consistent length across all histories
    min_len = min(len(state_history), len(fe_history), len(net_history))
    state_history = state_history[:min_len]
    fe_history = fe_history[:min_len]
    net_history = net_history[:min_len]

    print(f"Total steps: {min_len}")

    # Analyze both full trajectory and tail window
    tails = [("Full", 0), ("Tail (last 200)", -200)]
    
    for label, start_idx in tails:
        if abs(start_idx) > min_len:
            start_idx = 0  # Use full if tail window exceeds available data
            
        s_hist = state_history[start_idx:]
        f_hist = fe_history[start_idx:]
        n_hist = net_history[start_idx:]
        
        print(f"\n--- {label} Stats ---")

        # 1. Free Energy & Network Activations by State
        state_fe = {s: [] for s in STATE_MAP}
        state_net = {s: {n: [] for n in NETWORKS} for s in STATE_MAP}

        for i, state in enumerate(s_hist):
            if state in state_fe:
                state_fe[state].append(f_hist[i])
                for net in NETWORKS:
                    val = n_hist[i].get(net, 0.0)
                    state_net[state][net].append(val)

        print("[Free Energy & Network Activations]")
        for state, name in STATE_MAP.items():
            fe_vals = state_fe[state]
            avg_fe = np.mean(fe_vals) if fe_vals else 0.0
            
            net_avgs = []
            for net in NETWORKS:
                vals = state_net[state][net]
                avg_net = np.mean(vals) if vals else 0.0
                net_avgs.append(f"{net}: {avg_net:.2f}")
            
            print(f"  {name}: FE={avg_fe:.2f} | {', '.join(net_avgs)}")

        # 2. Dwell Times (contiguous state duration)
        dwell_times = {s: [] for s in STATE_MAP}
        if s_hist:
            current_state = s_hist[0]
            current_duration = 0
            for state in s_hist:
                if state == current_state:
                    current_duration += 1
                else:
                    if current_state in dwell_times:
                        dwell_times[current_state].append(current_duration)
                    current_state = state
                    current_duration = 1
            # Record final state duration
            if current_state in dwell_times:
                dwell_times[current_state].append(current_duration)

        print("[Dwell Times]")
        for state, name in STATE_MAP.items():
            durations = dwell_times[state]
            avg_dur = np.mean(durations) if durations else 0.0
            print(f"  {name}: {avg_dur:.1f} steps (n={len(durations)})")

        # 3. Meta-Awareness Statistics
        print("\n[Additional Diagnostics]")
        
        ma_hist = stats_data.get('meta_awareness_history', [])
        if start_idx != 0:
             ma_hist = ma_hist[start_idx:]
        else:
             ma_hist = ma_hist[:min_len]

        ma_mean = np.mean(ma_hist) if ma_hist else 0.0
        ma_sd = np.std(ma_hist) if ma_hist else 0.0
        print(f"  Meta-Awareness: Mean={ma_mean:.2f}, SD={ma_sd:.2f}")

        # 4. Network Variance & Coupling
        dan_vals = []
        fpn_vals = []
        all_net_vals = []
        
        for n_map in n_hist:
            dan = n_map.get('DAN', 0.0)
            fpn = n_map.get('FPN', 0.0)
            dan_vals.append(dan)
            fpn_vals.append(fpn)
            all_net_vals.extend(n_map.values())
            
        net_variance = np.var(all_net_vals)
        dan_fpn_corr = np.corrcoef(dan_vals, fpn_vals)[0, 1] if len(dan_vals) > 1 else 0.0
        
        print(f"  Overall Network Variance: {net_variance:.4f}")
        print(f"  DAN-FPN Correlation: {dan_fpn_corr:.2f}")

        # 5. Thoughtseed Dominance Distribution
        ts_hist = stats_data.get('dominant_ts_history', [])
        if start_idx != 0:
             ts_hist = ts_hist[start_idx:]
        else:
             ts_hist = ts_hist[:min_len]
             
        ts_counts = {}
        for ts in ts_hist:
            ts_counts[ts] = ts_counts.get(ts, 0) + 1
            
        print("  Thoughtseed Counts:")
        total_ts = len(ts_hist)
        for ts, count in ts_counts.items():
            print(f"    {ts}: {count} ({count/total_ts*100:.1f}%)")

        # 6. Attractor Dynamics Verification
        print("\n[Attractor Dynamics Verification]")
        
        act_hist = stats_data.get('activations_history', [])
        
        if start_idx != 0:
             act_hist = act_hist[start_idx:]
        else:
             act_hist = act_hist[:min_len]

        if not act_hist:
            print("  Error: No activations_history found.")
        else:
            try:
                # Thoughtseed indices: breath_focus=0, pain_discomfort=1, pending_tasks=2
                bf_vals = [step[0] for step in act_hist]
                pt_vals = [step[2] for step in act_hist]
                fe_vals = f_hist

                if bf_vals and pt_vals:
                    print(f"  Pending Tasks (y-axis):")
                    print(f"    Max: {max(pt_vals):.2f}")
                    print(f"    Mean: {np.mean(pt_vals):.2f}")
                    print(f"    > 0.8 count: {sum(1 for v in pt_vals if v > 0.8)} / {len(pt_vals)}")
                    
                    print(f"  Breath Focus (x-axis):")
                    print(f"    Min: {min(bf_vals):.2f}")
                    print(f"    Mean: {np.mean(bf_vals):.2f}")
                    
                    print(f"  Free Energy (Color/Z-axis):")
                    if fe_vals:
                        print(f"    Range: {min(fe_vals):.2f} - {max(fe_vals):.2f}")
                        print(f"    Mean: {np.mean(fe_vals):.2f}")
                    else:
                        print("    No Free Energy data.")
                    
                    # Validate attractor clustering claims
                    pt_under_05 = sum(1 for v in pt_vals if v < 0.5)
                    print(f"    Pending Tasks < 0.5: {pt_under_05} ({pt_under_05/len(pt_vals)*100:.1f}%)")
            except Exception as e:
                print(f"  Error parsing activations: {e}")


if __name__ == "__main__":
    calculate_stats("novice")
    calculate_stats("expert")
