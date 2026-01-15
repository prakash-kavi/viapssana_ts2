"""Run simulation using the variant trainer for regression testing."""
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import numpy as np
from meditation_model import ActInfAgent
from meditation_trainer_variant import TrainerVariant


def run_variant():
    seed = 42
    T = 1000
    logging.info("Starting variant simulation...")
    learner_novice = ActInfAgent(experience_level='novice', timesteps_per_cycle=T)
    TrainerVariant(learner_novice).train(save_outputs=False, seed=seed)
    logging.info("Novice natural transitions: %d", learner_novice.natural_transition_count)

    learner_expert = ActInfAgent(experience_level='expert', timesteps_per_cycle=T)
    TrainerVariant(learner_expert).train(save_outputs=False, seed=seed)
    logging.info("Expert natural transitions: %d", learner_expert.natural_transition_count)

if __name__ == '__main__':
    run_variant()
