"""
generate_all_plots.py

Run all viz generation functions (convergence, attractors, diagnostics).
"""
from pathlib import Path
import os
import logging

from . import plotting_utils as pu
from . import plot_diagnostics as pd

def main():
    os.makedirs(pu.PLOT_DIR, exist_ok=True)

    # Use tail-windowed stats(last 200 timesteps) for
    # time-series/hierarchy plots and full stats for aggregate
    # distributions (FE bars, dwell times). Convergence plots use full series(1000 timesteps).
    nov_ts, nov_ai, nov_stats = pu.load_json_data('novice')
    exp_ts, exp_ai, exp_stats = pu.load_json_data('expert')

    # Tail-sliced copies for visualization panels that focus on recent activity
    nov_tail = pu.get_tail_stats(nov_stats)
    exp_tail = pu.get_tail_stats(exp_stats)
    nov_tail['experience_level'] = 'novice'
    exp_tail['experience_level'] = 'expert'

    # Hierarchy and combined time-series: use tail window (last 200 timesteps)
    pd.plot_hierarchy(nov_tail, save_path=os.path.join(pu.PLOT_DIR, 'Fig4A_Hierarchy_Novice.png'))
    pd.plot_hierarchy(exp_tail, save_path=os.path.join(pu.PLOT_DIR, 'Fig4B_Hierarchy_Expert.png'))
    pd.plot_time_series(nov_tail, exp_tail, save_path=os.path.join(pu.PLOT_DIR, 'FigS1_TimeSeries.png'))

    # Free energy bar, network radar and dwell times: use full statistics
    pd.plot_free_energy_bar(nov_stats, exp_stats, save_path=os.path.join(pu.PLOT_DIR, 'Fig3A_FEBar.png'))
    pd.plot_network_radar(nov_ai, exp_ai, save_path=os.path.join(pu.PLOT_DIR, 'Fig3B_Radar.png'))
    pd.plot_dwell_times(nov_stats, exp_stats, save_path=os.path.join(pu.PLOT_DIR, 'Fig3C_Dwell.png'))

    logging.info("Saved diagnostic plots to %s", pu.PLOT_DIR)

if __name__ == '__main__':
    main()
