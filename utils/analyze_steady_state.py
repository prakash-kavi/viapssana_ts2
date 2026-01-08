"""
analyze_steady_state.py

Analyzes the convergence to steady-state behavior by examining state distributions
over the final window of training timesteps. Helps validate that the simulation
has reached stable dynamics before using the data for analysis.

Usage:
    python analyze_steady_state.py
"""

import json
import numpy as np


def analyze_last_200_steps(filename, total_steps=1000, window=200):
    """
    Analyze state distribution over the last N timesteps.
    
    Args:
        filename: Path to transition_stats_*.json file
        total_steps: Total simulation timesteps (default 1000)
        window: Number of final steps to analyze (default 200)
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    timestamps = data['transition_timestamps']
    patterns = data['state_transition_patterns']
    
    # Reconstruct full state history from transition patterns
    # Timestamps mark the END of each state dwell period
    state_history = []
    last_time = 0
    
    for i, pattern in enumerate(patterns):
        # Pattern: (current_state, next_state, ts_acts, net_acts, vfe)
        state = pattern[0]
        
        if i < len(timestamps):
            end_time = timestamps[i]
            duration = end_time - last_time
            
            # Fill history with current state for its duration
            state_history.extend([state] * int(duration))
            last_time = end_time
            
    # Fill remaining timesteps with final state if needed
    if len(state_history) < total_steps:
        last_state = patterns[-1][1] if patterns else "breath_control"
        state_history.extend([last_state] * (total_steps - len(state_history)))
        
    # Extract the final window
    relevant_history = state_history[-window:]
    
    # Calculate state distribution
    counts = {}
    for state in relevant_history:
        counts[state] = counts.get(state, 0) + 1
        
    print(f"Analysis for {filename} (Last {window} steps):")
    total = len(relevant_history)
    for state, count in counts.items():
        fraction = count / total
        print(f"  State: {state:<20} Count: {count:<5} Fraction: {fraction:.2%}")


if __name__ == "__main__":
    print("--- NOVICE ---")
    analyze_last_200_steps('../data/transition_stats_novice.json')
    print("\n--- EXPERT ---")
    analyze_last_200_steps('../data/transition_stats_expert.json')
