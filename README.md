# Vipassana-TS2: Thoughtseeds Framework Implementation

This repository contains the reference implementation of the **Thoughtseeds Framework** for modeling Vipassana meditation dynamics using Active Inference.

## Project Structure

The codebase is organized for reproducibility and clarity:

*   **`meditation_model.py`**: Core implementation of the Active Inference learner and the Thoughtseeds cognitive architecture.
*   **`meditation_config.py`**: Centralized configuration for simulation parameters (priors, network profiles, thoughtseed definitions).
*   **`run_simulation.py`**: Driver script to execute the simulation for Novice and Expert profiles and generate data.
*   **`plotting_utils.py`**: Shared utilities for data loading and publication-quality plotting styles.
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
| **Figure 3** | Reference Profile Diagnostics (Tail Window) | `python plot_diagnostics.py` | `plots/Fig3_*.png` |
| **Figure 4** | Hierarchical Dynamics Snapshots | `python plot_hierarchy.py` | `plots/Fig4_*.png` |
| **Figure 5** | Attractor Landscapes | `python plot_attractors.py` | `plots/Fig5_*.png` |
| **Figure S1** | Convergence Diagnostics (Supplementary) | `python plot_convergence.py` | `plots/FigS1_*.png` |

## Data Files

*   `active_inference_params_*.json`: Learned parameters (precision, complexity, etc.) and network expectations.
*   `transition_stats_*.json`: Statistics on state transitions and dwell times.
*   `thoughtseed_params_*.json`: Thoughtseed parameters and full time-series history.
