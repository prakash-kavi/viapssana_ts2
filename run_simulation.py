"""
run_simulation.py

Main entry point for running the meditation simulation.
This script initializes and trains the Active Inference models for both novice and expert cohorts.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import numpy as np
from meditation_model import ActInfLearner
from meditation_utils import ensure_directories

def run_simulation():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    T = 1000  # Total timesteps for training   
    
    logging.info("Starting simulation...")
    ensure_directories()
    
    # Create and train active inference learners
    logging.info("--- Training Novice Model ---")
    learner_novice = ActInfLearner(experience_level='novice', timesteps_per_cycle=T)
    learner_novice.train()

    logging.info("--- Training Expert Model ---")
    learner_expert = ActInfLearner(experience_level='expert', timesteps_per_cycle=T)
    learner_expert.train()
    
    logging.info("Active Inference training completed for both novice and expert models.")

if __name__ == "__main__":
    run_simulation()
