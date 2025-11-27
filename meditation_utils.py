"""
meditation_utils.py

Utility functions for the Vipassana Entropy meditation simulation.
"""

import os
import logging
import numpy as np
import json
from meditation_config import THOUGHTSEED_AGENTS

def ensure_directories(base_dir=None):
    """Create iwai/data and iwai/plots by default (package-relative when base_dir is None)."""
    if not base_dir:
        base_dir = os.path.dirname(__file__)
        
    base_dir = os.fspath(base_dir)
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logging.info("Directories created/verified: data -> %s, plots -> %s", data_dir, plots_dir)

def _save_json_outputs(learner, output_dir=None):
    """Save JSON outputs"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")
    logging.info("Generating consumer-ready JSON files...")

    # Helper function to convert numpy arrays to lists
    def convert_numpy_to_lists(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_lists(i) for i in obj]
        else:
            return obj
    
    # 1. ThoughtseedNetwork parameters with network profiles
    thoughtseed_params = {
        "interactions": THOUGHTSEED_AGENTS,
        "agent_parameters": {
            ts: {
                "base_activation": float(np.mean([act[i] for act in learner.activations_history])),
                "responsiveness": float(max(0.5, 1.0 - np.std([act[i] for act in learner.activations_history]))),
                "decay_rate": THOUGHTSEED_AGENTS[ts]["decay_rate"],
                "recovery_rate": THOUGHTSEED_AGENTS[ts]["recovery_rate"],
                "network_profile": learner.learned_network_profiles["thoughtseed_contributions"][ts]
            } for i, ts in enumerate(learner.thoughtseeds)
        },
        "activation_means_by_state": {
            state: {
                ts: float(np.mean([
                    learner.activations_history[j][i] 
                    for j, s in enumerate(learner.state_history) if s == state
                ])) for i, ts in enumerate(learner.thoughtseeds)
            } for state in learner.states if any(s == state for s in learner.state_history)
        },
        "network_activations_by_state": {
            state: {
                net: float(np.mean([
                    learner.network_activations_history[j][net]
                    for j, s in enumerate(learner.state_history) if s == state
                ])) for net in learner.networks
            } for state in learner.states if any(s == state for s in learner.state_history)
        },
        "learned_network_profiles": convert_numpy_to_lists(learner.learned_network_profiles)
    }
    
    # Add time series data (converting NumPy arrays to lists)
    thoughtseed_params["time_series"] = {
        "activations_history": convert_numpy_to_lists(learner.activations_history),
        "network_activations_history": convert_numpy_to_lists(learner.network_activations_history),
        "meta_awareness_history": learner.meta_awareness_history,  
        "free_energy_history": learner.free_energy_history,  
        "state_history": learner.state_history,
        "dominant_ts_history": learner.dominant_ts_history  

    }
    
    out_path_ts = os.path.join(output_dir, f"thoughtseed_params_{learner.experience_level}.json")
    with open(out_path_ts, "w", encoding="utf-8") as f:
        json.dump(thoughtseed_params, f, indent=2)
    
    # 2. Active Inference parameters
    active_inf_params = {
        "precision_weight": learner.precision_weight,
        "complexity_penalty": learner.complexity_penalty,
        "learning_rate": learner.learning_rate,
        "average_free_energy_by_state": {
            state: float(np.mean([
                learner.free_energy_history[j]
                for j, s in enumerate(learner.state_history) if s == state
            ])) for state in learner.states if any(s == state for s in learner.state_history)
        },
        "average_prediction_error_by_state": {
            state: float(np.mean([
                learner.prediction_error_history[j]
                for j, s in enumerate(learner.state_history) if s == state
            ])) for state in learner.states if any(s == state for s in learner.state_history)
        },
        "average_precision_by_state": {
            state: float(np.mean([
                learner.precision_history[j]
                for j, s in enumerate(learner.state_history) if s == state
            ])) for state in learner.states if any(s == state for s in learner.state_history)
        },
        "network_expectations": learner.learned_network_profiles["state_network_expectations"]
    }
    
    out_path_ai = os.path.join(output_dir, f"active_inference_params_{learner.experience_level}.json")
    with open(out_path_ai, "w", encoding="utf-8") as f:
        json.dump(active_inf_params, f, indent=2)

    logging.info("  - JSON parameter files saved to %s directory", output_dir)
