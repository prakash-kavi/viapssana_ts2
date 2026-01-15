"""Run multiple variant trainers to localize behavioral change."""
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import numpy as np
from meditation_model import ActInfAgent

from meditation_trainer_variant import TrainerVariant
from meditation_trainer_variant_dupinit import TrainerDupInit
from meditation_trainer_variant_hardseq import TrainerHardSeq


def run_tests():
    seed = 42
    T = 1000

    # Variant: both changes
    learner = ActInfAgent(experience_level='novice', timesteps_per_cycle=T)
    TrainerVariant(learner).train(save_outputs=False, seed=seed)
    logging.info('Variant (both) novice transitions: %d', learner.natural_transition_count)

    # Variant: duplicate init only
    learner = ActInfAgent(experience_level='novice', timesteps_per_cycle=T)
    TrainerDupInit(learner).train(save_outputs=False, seed=seed)
    logging.info('Variant (dupinit) novice transitions: %d', learner.natural_transition_count)

    # Variant: hard seq only
    learner = ActInfAgent(experience_level='novice', timesteps_per_cycle=T)
    TrainerHardSeq(learner).train(save_outputs=False, seed=seed)
    logging.info('Variant (hardseq) novice transitions: %d', learner.natural_transition_count)

    # Current Trainer (import from meditation_trainer)
    from meditation_trainer import Trainer
    learner = ActInfAgent(experience_level='novice', timesteps_per_cycle=T)
    Trainer(learner).train(save_outputs=False, seed=seed)
    logging.info('Current trainer novice transitions: %d', learner.natural_transition_count)

if __name__ == '__main__':
    run_tests()
