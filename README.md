# Vipassana-TS2: Thoughtseeds Framework Implementation

This repository contains the reference implementation of the **Thoughtseeds Framework** for modeling Vipassana meditation dynamics.

**Important Note:** This project is a **computational simulation of a dynamical system** that implements **stochastic dynamics with coupled Ornstein-Uhlenbeck processes** within an Active Inference formulation. It is a theoretical model designed to explore cognitive architectures and **does not model real-world empirical data**.

It demonstrates the architectural utility of the "Thoughtseeds" framework as a scaffold for linking subjective phenomenology with measurable neural dynamics. The three-level hierarchical structure (attentional networks â†’ thoughtseeds â†’ meta-cognition) captures the nested organization of the meditative mind, replicating complex expertâ€“novice phenomenological differences through the minimization of Variational Free Energy (VFE).

This implementation is a significant revision of the earlier work presented at [IWAI 2025](https://iwaiworkshop.github.io/): https://github.com/prakash-kavi/aif_iwai2025_thoughtseeds

## Conceptual Framework

![Figure 1](Mediative_cycle.jpg)

**Figure 1. The four canonical stages of Focussed Attention meditation**
Breath Focus, Mind Wandering, Meta-Awareness, and Redirect Attention.

![Figure 2](Thoughtseeds%20Framework.jpg)

**Figure 2. Thoughtseeds Framework**
Three-level hierarchical organization of thoughtseeds framework, rooted in Active Inference and Global Workspace Theory (GWT).

*   **Level 1: Attentional Network Substrate** comprises four simplified attentional networks (Yeo et al., 2011)â€”DMN, VAN, DAN, and FPN. These networks are substrates, i.e., large-scale functional ensembles of neuronal packets at nested scales that provide the context for finer-grained dynamics.
*   **Level 2: Thoughtseed Network** consists of attentional agents (thoughtseeds), such as breath_focus, pending_tasks, or self_reflection, which emerge from coordinated activity across the Level 1 ensembles. Thoughtseeds are hypothesized to represent specific contents of thought or attention, and compete for dominance in the Global Workspace via a simplified winner-takes-all dynamics (Baars 1997; Dehaene and Changeux, 2011).
*   **Level 3: Meta-cognition** regulates precision weighting and meta-awareness, gating which thoughtseeds are amplified or suppressed to align behavior with meditative goals.

The system operates by minimizing **Variational Free Energy (VFE)**, where the agent (meditator) attempts to align their internal model (expectations of focus) with sensory states (current thoughtseeds). Novices exhibit higher VFE and volatility due to weaker priors and lower precision, while Experts demonstrate "Equanimity" through optimized precision weighting and efficient error minimization.

## Project Structure

The codebase is organized for reproducibility and clarity:

*   **`meditation_model.py`**: Core implementation of the `ActInfLearner` class, implementing the Active Inference loop (Perception -> Action -> Learning).
*   **`meditation_config.py`**: Centralized configuration for simulation parameters, defining specific profiles for Novice and Expert meditators (priors, network profiles, transition thresholds).
*   **`run_simulation.py`**: Driver script to execute the simulation for Novice and Expert profiles and generate data.
*   **`viz/plotting_utils.py`**: Shared utilities for data loading and publication-quality plotting styles.
*   **`viz/plot_diagnostics.py`**: Generates time-series and distribution plots (Figure 3).
*   **`viz/plot_attractors.py`**: Generates 2D and 3D attractor landscape visualizations (Figure 5).
*   **`viz/plot_convergence.py`**: Generates convergence metrics (Figure S1).
*   **`data/`**: Contains the generated JSON outputs from the simulation.
*   **`plots/`**: Contains the generated figures used in the manuscript.

## Reproducing the Results

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Simulation
To generate the data (Novice and Expert profiles):
```bash
python run_simulation.py
```
This will populate the `data/` directory with JSON files containing training history and stabilized parameters.

### 3. Generate Figures
The figures in the manuscript can be reproduced using the following scripts:

| Figure | Description | Script | Output |
| :--- | :--- | :--- | :--- |
| **Figures 3 & 4** | Reference Profile Diagnostics & Hierarchical Dynamics | `python -m viz.plot_diagnostics` | `plots/Fig3_*.png, plots/Fig4_*.png` |
| **Figure 5** | Attractor Landscapes | `python -m viz.plot_attractors` | `plots/Fig5_*.png` |
| **Figure S1** | Convergence Diagnostics (Supplementary) | `python -m viz.plot_convergence` | `plots/FigS1_*.png` |

## Data Files

*   `active_inference_params_*.json`: Learned parameters (precision, complexity, etc.) and network expectations.
*   `transition_stats_*.json`: Statistics on state transitions and dwell times.
*   `thoughtseed_params_*.json`: Thoughtseed parameters and full time-series history.

## Validation Utilities

The `utils/` directory contains diagnostic scripts for validating simulation outputs:

*   **`analyze_steady_state.py`**: Analyzes convergence to steady-state by examining state distributions over the final 200 timesteps. Helps verify that the simulation has reached stable dynamics before using data for analysis.
*   **`verify_stats.py`**: Comprehensive validation tool that computes Free Energy, network activations, dwell times, meta-awareness statistics, and attractor dynamics metrics for both novice and expert profiles. Useful for cross-checking manuscript claims against simulation outputs.

Run from the project root:
```bash
python -m utils.analyze_steady_state
python -m utils.verify_stats
```


