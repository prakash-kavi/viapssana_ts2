"""
meditation_model.py

Implements core meditation models:
1. `AgentConfig`: Foundation class providing stochastic thoughtseed dynamics and state transitions.
2. `ActInfAgent`: Active Inference extension implementing the three-level framework.
"""

import numpy as np
import os
import json
import logging
from collections import defaultdict
import copy

from meditation_config import (
    THOUGHTSEEDS, STATES, STATE_DWELL_TIMES,
    ActiveInferenceConfig, ThoughtseedParams, MetacognitionParams,
    NETWORK_PROFILES, DEFAULTS
)
from meditation_utils import ensure_directories, _save_json_outputs, ou_update, clip_array

class AgentConfig:
    """
    Foundation class for thoughtseed dynamics.
    Provides core methods for behavior, meta-awareness, and state transitions.
    """
    
    def __init__(self, experience_level='novice', timesteps_per_cycle=200):
        self.experience_level = experience_level
        self.timesteps = timesteps_per_cycle
        self.thoughtseeds = THOUGHTSEEDS
        self.states = STATES
        self.num_thoughtseeds = len(self.thoughtseeds)
        
        # State tracking
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.natural_transition_count = 0
        self.forced_transition_count = 0
        
        # History tracking
        self.activations_history = []
        self.state_history = []
        self.meta_awareness_history = []
        self.dominant_ts_history = []
        self.state_history_over_time = []
        
        # Get noise level from config
        aif_params = ActiveInferenceConfig.get_params(experience_level)
        self.noise_level = aif_params['noise_level']
        
        # Track activation patterns at transition points
        self.transition_activations = {state: [] for state in self.states}
        
        # Track distraction buildup patterns
        self.distraction_buildup_rates = []
        
        # Initialize accumulators (Leaky Integrators)
        self.dmn_accumulator = 0.0
        self.fpn_accumulator = 0.0
        # Placeholder for previous network activations (initialized in ActInfAgent)
        self.prev_network_acts = None

    def get_target_activations(self, state, meta_awareness):
        # Return target activations for `state` (modulated by `meta_awareness`)
        targets_dict = ThoughtseedParams.get_target_activations(
            state, meta_awareness, self.experience_level)
        
        target_activations = np.zeros(self.num_thoughtseeds)
        for i, ts in enumerate(self.thoughtseeds):
            target_activations[i] = targets_dict[ts]
        
        # Add noise for biological plausibility
        target_activations += np.random.normal(0, self.noise_level, size=self.num_thoughtseeds)
        return np.clip(target_activations, 0.05, 1.0)

    def get_dwell_time(self, state):
        # Random dwell time bounded by STATE_DWELL_TIMES and minimal biological plausibility
        config_min, config_max = STATE_DWELL_TIMES[self.experience_level][state]
        
        # Ensure minimal biological plausibility
        if state in ['meta_awareness', 'redirect_breath']:
            min_biological = 1
            max_biological = config_max
        else:
            min_biological = 3
            max_biological = config_max
        
        return max(min_biological, min(max_biological, np.random.randint(config_min, config_max + 1)))

    def get_meta_awareness(self, current_state, activations):
        # Compute meta-awareness from thoughtseed activations and state
        activations_dict = {ts: activations[i] for i, ts in enumerate(self.thoughtseeds)}
        
        return MetacognitionParams.calculate_meta_awareness(
            state=current_state,
            thoughtseed_activations=activations_dict,
            experience_level=self.experience_level
        )

class ActInfAgent(AgentConfig):
    """
    Active Inference extension implementing the three-level Thoughtseeds Framework:
    1. Attentional Network Ensembles (DMN, VAN, DAN, FPN)
    2. Thoughtseed Dynamics
    3. Metacognitive Regulation
    
    Models the Vipassana cycle by minimizing variational free energy.
    """
    def __init__(self, experience_level='novice', timesteps_per_cycle=200):
        super().__init__(experience_level, timesteps_per_cycle)
        
        self.networks = ['DMN', 'VAN', 'DAN', 'FPN']
        # Initialize prev_network_acts to a zeroed dict so downstream methods
        # can rely on the attribute existing and check truthiness.
        self.prev_network_acts = {net: 0.0 for net in self.networks}
        self.network_activations_history = []
        self.free_energy_history = []
        self.prediction_error_history = []
        self.precision_history = []
        
        # Get all active inference parameters from centralized config
        aif_params = ActiveInferenceConfig.get_params(experience_level)
        
        self.precision_weight = aif_params['precision_weight']
        self.complexity_penalty = aif_params['complexity_penalty']
        self.learning_rate = aif_params['learning_rate']
        self.noise_level = aif_params['noise_level']
        self.memory_factor = aif_params['memory_factor']
        self.fpn_enhancement = aif_params['fpn_enhancement']
        self.fpn_reflection_value = aif_params.get('fpn_reflection_value', 0.15)
        self.fpn_equanimity_value = aif_params.get('fpn_equanimity_value', 0.2)
        self.transition_thresholds = aif_params['transition_thresholds']
        self.distraction_pressure = aif_params['distraction_pressure']
        self.fatigue_rate = aif_params['fatigue_rate']
        self.softmax_temperature = aif_params['softmax_temperature']
        self.fatigue_threshold = aif_params['fatigue_threshold']
        
        # Track learned network profiles
        self.learned_network_profiles = {
            "thoughtseed_contributions": {ts: {} for ts in self.thoughtseeds},
            "state_network_expectations": {state: {} for state in self.states}
        }
        
        # Initialize tracking variables (already defined on AgentConfig)
        
        self.in_transition = False
        self.transition_counter = 0
        self.transition_target = None
        self.prev_meta_awareness = 0.0
        
        # Initialize with default profiles
        for ts in self.thoughtseeds:
            self.learned_network_profiles["thoughtseed_contributions"][ts] = NETWORK_PROFILES["thoughtseed_contributions"][ts].copy()

        for state in self.states:
            self.learned_network_profiles["state_network_expectations"][state] = NETWORK_PROFILES["state_expected_profiles"][state][self.experience_level].copy()
            
    def compute_network_activations(self, thoughtseed_activations, current_state, meta_awareness, dt=1.0):
        # Compute attentional network activations from thoughtseed activations and state
        # Get thoughtseed-to-network contribution matrix
        ts_to_network = self.learned_network_profiles["thoughtseed_contributions"]
        
        # 1. Calculate Base Targets
        target_acts = {net: DEFAULTS['NETWORK_BASE'] for net in self.networks}
        
        # Bottom-up: Thoughtseeds -> Networks
        for i, ts in enumerate(self.thoughtseeds):
            ts_act = thoughtseed_activations[i]
            for net in self.networks:
                target_acts[net] += ts_act * ts_to_network[ts][net]
    
        # Top-down: State Expectations -> Networks
        state_expect = self.learned_network_profiles["state_network_expectations"][current_state]
        for net in self.networks:
            # Meta-awareness amplifies top-down control
            # Experts maintain control with less effort (Neural Efficiency)
            meta_factor = meta_awareness * (1.05 if self.experience_level == 'expert' else 1.0)
            state_influence = state_expect[net] * meta_factor
            target_acts[net] = 0.5 * target_acts[net] + 0.5 * state_influence

        # 2. Apply Coupled Dynamics        
        if self.prev_network_acts:
            current_dan = self.prev_network_acts['DAN']
            current_fpn = self.prev_network_acts['FPN']
            
            # A. Neural Efficiency (FPN Regulation)
            # Models "effortless focus": If DAN is high (stable), FPN relaxes.
            if current_state in ['breath_control', 'redirect_breath']:
                focus_error = max(0, 0.9 - current_dan)
                fpn_demand = DEFAULTS['FPN_BASE_DEMAND'] + (DEFAULTS['FPN_FOCUS_MULT'] * focus_error)

                # Experts relax control faster when stable
                efficiency_weight = DEFAULTS['EFFICIENCY_WEIGHT_EXPERT'] if self.experience_level == 'expert' else DEFAULTS['EFFICIENCY_WEIGHT_NOVICE']
                target_acts['FPN'] = (1 - efficiency_weight) * target_acts['FPN'] + efficiency_weight * fpn_demand

            # B. FPN Drives DAN + DAN Hysteresis
            # FPN Drive: Top-down recruitment with saturation kinetics
            target_acts['DAN'] += DEFAULTS['FPN_TO_DAN_GAIN'] * current_fpn * (1.0 - current_dan)
            
            # DAN Hysteresis: Momentum of sustained attention; disrupted by DMN
            current_dmn = self.prev_network_acts.get('DMN', 0)
            hysteresis_strength = DEFAULTS['HYSTERESIS_EXPERT'] if self.experience_level == 'expert' else DEFAULTS['HYSTERESIS_NOVICE']
            target_acts['DAN'] += hysteresis_strength * current_dan * (1.0 - current_dmn)

        # C. Mutual Inhibition (Task-Positive vs Task-Negative)
        anticorrelation_force = DEFAULTS['ANTICORRELATION_FORCE']
        if self.prev_network_acts:
            target_acts['DAN'] -= anticorrelation_force * self.prev_network_acts['DMN'] * target_acts['DAN']
            target_acts['DMN'] -= anticorrelation_force * self.prev_network_acts['DAN'] * target_acts['DMN']
        
        # VAN Leaky Integrator (The Detector)
        if self.prev_network_acts:
            current_dmn = self.prev_network_acts['DMN']
            self.dmn_accumulator = 0.9 * self.dmn_accumulator + 0.1 * current_dmn
            
            # Fire if threshold crossed (Refractory period via reset)
            if self.dmn_accumulator > DEFAULTS.get('VAN_TRIGGER', 0.7):
                target_acts['VAN'] += DEFAULTS['VAN_SPIKE']  # Salience spike
                self.dmn_accumulator = 0.0 # Reset

        # D. FPN Accumulator (Cognitive Fatigue)
        if self.prev_network_acts:
            current_fpn = self.prev_network_acts['FPN']
            
            # Accumulate FPN usage (Leaky Integrator)
            self.fpn_accumulator = DEFAULTS['FPN_ACCUM_DECAY'] * self.fpn_accumulator + DEFAULTS['FPN_ACCUM_INC'] * current_fpn
            
            # Fatigue Threshold: When effort exceeds capacity, focus collapses.
            if self.fpn_accumulator > self.fatigue_threshold:
                target_acts['DAN'] *= DEFAULTS.get('FPN_COLLAPSE_DAN_MULT', 0.6) # Collapse focus
                target_acts['DMN'] += DEFAULTS.get('FPN_COLLAPSE_DMN_INC', 0.2) # Surge distraction
                self.fpn_accumulator = DEFAULTS.get('FATIGUE_RESET', 0.4) # Reset

        # 3. Ornstein-Uhlenbeck Process (Smoothing & Stability)
        current_acts = {}
        
        if self.prev_network_acts:
            # Theta (θ): Mean reversion rate. Derived from memory_factor.
            theta = 1.0 - self.memory_factor
            sigma = self.noise_level
            
            for net in self.networks:
                x_prev = self.prev_network_acts[net]
                mu = target_acts[net]

                current_acts[net] = float(ou_update(x_prev, mu, theta, sigma, dt))
        else:
            # First step initialization
            current_acts = target_acts
            
        # Normalize
        from meditation_utils import clip_array

        for net in self.networks:
            current_acts[net] = float(clip_array(current_acts[net], DEFAULTS['NETWORK_CLIP_MIN'], DEFAULTS['NETWORK_CLIP_MAX']))

        # VAN values > neurophysiological cap
        max_van = DEFAULTS['VAN_MAX']
        if current_acts['VAN'] > max_van:
            current_acts['VAN'] = max_van
            
        return current_acts
    
    def get_sensory_inference(self, network_acts):
        # Infer thoughtseed beliefs from network activations using learned profiles
        ts_contribs = self.learned_network_profiles["thoughtseed_contributions"]
        inferred = np.zeros(self.num_thoughtseeds)
        
        for i, ts in enumerate(self.thoughtseeds):
            match_score = 0.0
            total_weight = 0.0
            
            for net in self.networks:
                # The weight represents P(network | thoughtseed)
                weight = ts_contribs[ts][net]
                
                # We want to know P(thoughtseed | network)
                match_score += network_acts[net] * weight
                total_weight += weight
            
            if total_weight > 0:
                inferred[i] = match_score / total_weight
            else: # pragma: no cover
                inferred[i] = 0.1
                
        return clip_array(inferred, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

    def calculate_vfe(self, current_seeds, prior_seeds, sensory_inference, meta_awareness, vfe_trend=0.0):
        # Compute variational free energy and components (sensory/prior NLL)
        # 1. Accuracy (Sensory NLL): Divergence between Beliefs and Reality
        # NLL(o|s) ~ KL(o||s)
        sensory_nll = np.sum(
            sensory_inference * np.log(sensory_inference / (current_seeds + 1e-9)) + 
            (1 - sensory_inference) * np.log((1 - sensory_inference) / (1 - current_seeds + 1e-9))
        )
        
        # 2. Complexity (Prior NLL): Divergence between Beliefs and Policy
        prior_nll = np.sum(
            current_seeds * np.log(current_seeds / (prior_seeds + 1e-9)) + 
            (1 - current_seeds) * np.log((1 - current_seeds) / (1 - prior_seeds + 1e-9))
        )
        
        # 3. Precisions
        # Dynamic Precision Modulation: If VFE is dropping, increase precision (Confidence)
        precision_mod = np.clip(-1.0 * vfe_trend, -0.3, 0.3)

        # Lucidity (Sensory Precision): Increases with VAN (Insight)
        van_proxy = sensory_inference[self.thoughtseeds.index('self_reflection')]
        pi_sensory = (0.1 + (5.0 * van_proxy)) * (1.0 + precision_mod) * self.precision_weight
        
        # Attention (Prior Precision): Modulated by Meta-Awareness
        pi_prior = (1.0 + (3.0 * meta_awareness)) * (1.0 + precision_mod) * self.complexity_penalty
        
        # Total VFE
        vfe = (sensory_nll * pi_sensory) + (prior_nll * pi_prior)
        
        return vfe, sensory_nll, prior_nll
   
    def update_network_profiles(self, thoughtseed_activations, network_activations, current_state, prediction_errors):
        # Update learned mapping from thoughtseeds -> networks using prediction errors
        # Only update after some initial observations
        if len(self.network_activations_history) < 10:
            return
        
        for i, ts in enumerate(self.thoughtseeds):
            ts_act = thoughtseed_activations[i]  # z_i(t)
            
            if ts_act > 0.2:
                for net in self.networks:
                    current_error = prediction_errors[net] # δ_k(t)
                    
                    # Calculate precision (confidence)
                    precision = 1.0 + (5.0 if self.experience_level == 'expert' else 2.0) * len(self.network_activations_history)/self.timesteps
                    
                    # Bayesian-inspired update (Eq 3)
                    error_sign = 1 if network_activations[net] < self.learned_network_profiles["state_network_expectations"][current_state][net] else -1
                    update = self.learning_rate * (error_sign * current_error) * ts_act / precision
                    
                    # Update contribution
                    self.learned_network_profiles["thoughtseed_contributions"][ts][net] += update
                    
                    # Bound weights [W_min, W_max]
                    self.learned_network_profiles["thoughtseed_contributions"][ts][net] = np.clip(
                        self.learned_network_profiles["thoughtseed_contributions"][ts][net], 0.1, 0.9)
        
        # Update state network expectations (slower learning rate)
        slow_rate = self.learning_rate * 0.3
        for net in self.networks:
            current_expect = self.learned_network_profiles["state_network_expectations"][current_state][net]
            new_value = (1 - slow_rate) * current_expect + slow_rate * network_activations[net]
            self.learned_network_profiles["state_network_expectations"][current_state][net] = new_value
    
    def get_network_modulation(self, network_acts, current_state):
        # Compute adjustments to thoughtseed targets driven by network activity
        modulations = {ts: 0.0 for ts in self.thoughtseeds}
        
        # DMN enhances pending_tasks and self_reflection, suppresses breath_focus
        dmn_strength = network_acts.get('DMN', 0)

        modulations['pending_tasks'] += DEFAULTS.get('DMN_PENDING_VALUE', 0.15) * dmn_strength
        modulations['self_reflection'] += DEFAULTS.get('DMN_REFLECTION_VALUE', 0.05) * dmn_strength
        modulations['breath_focus'] -= DEFAULTS.get('DMN_BREATH_VALUE', 0.2) * dmn_strength

        # VAN enhances pain_discomfort (salience) and self_reflection during meta_awareness
        van_strength = network_acts.get('VAN', 0)

        modulations['pain_discomfort'] += DEFAULTS.get('VAN_PAIN_VALUE', 0.15) * van_strength

        if current_state == "meta_awareness":
            modulations['self_reflection'] += DEFAULTS.get('VAN_REFLECTION_VALUE', 0.2) * van_strength

        # DAN enhances breath_focus, suppresses distractions
        dan_strength = network_acts.get('DAN', 0)

        modulations['breath_focus'] += DEFAULTS.get('DAN_BREATH_VALUE', 0.2) * dan_strength
        modulations['pending_tasks'] -= DEFAULTS.get('DAN_PENDING_VALUE', 0.15) * dan_strength
        modulations['pain_discomfort'] -= DEFAULTS.get('DAN_PAIN_VALUE', 0.1) * dan_strength
        
        # FPN enhances self_reflection and equanimity (metacognition and regulation)
        fpn_strength = network_acts.get('FPN', 0)
        fpn_enhancement = self.fpn_enhancement

        modulations['self_reflection'] += self.fpn_reflection_value * fpn_strength * fpn_enhancement
        modulations['equanimity'] += self.fpn_equanimity_value * fpn_strength * fpn_enhancement
                
        return modulations

    def get_transition_probabilities(self, activations, network_acts):
                # Score and normalize possible next states by similarity to learned profiles
        scores = {}
        temp = max(1e-6, getattr(self, 'softmax_temperature', 0.5))

        # Use previous meta-awareness as a proxy (fallback to 0.5)
        meta = getattr(self, 'prev_meta_awareness', 0.5) if getattr(self, 'prev_meta_awareness', None) is not None else 0.5

        for state in self.states:
            # Network expectation similarity (negative L2 distance)
            expect = self.learned_network_profiles["state_network_expectations"][state]
            expect_vec = np.array([expect[net] for net in self.networks])
            net_vec = np.array([network_acts.get(net, 0.0) for net in self.networks])
            net_dist = np.linalg.norm(net_vec - expect_vec)
            net_score = np.exp(-net_dist / (temp * 1.0))

            # Activation similarity: compare to target activations for that state
            try:
                target_ts = self.get_target_activations(state, meta)
                act_dist = np.linalg.norm(activations - target_ts)
                act_score = np.exp(-act_dist / (temp * 1.0))
            except Exception:
                act_score = 1.0

            # Combine scores (multiplicative fusion)
            scores[state] = float(net_score * act_score)

        # Normalize into probabilities (avoid division by zero)
        total = sum(scores.values())
        if total <= 0:
            # Uniform over states as a safe fallback
            n = len(self.states)
            return {s: 1.0 / n for s in self.states}

        return {s: v / total for s, v in scores.items()}

    def update_thoughtseed_dynamics(self, current_activations, target_activations, current_state, current_dwell, dwell_limit):
        # Evolve thoughtseed activations toward targets using OU dynamics
        dt = DEFAULTS['DEFAULT_DT']
        updated_activations = current_activations.copy()
        
        # 1. Calculate Dynamic Targets (mu)
        mu = np.array(target_activations)
        
        # Apply Network Modulation to Targets
        if self.prev_network_acts:
            modulations = self.get_network_modulation(self.prev_network_acts, current_state)
            for i, ts in enumerate(self.thoughtseeds):
                mu[i] += modulations[ts]
        
        # Apply dynamic modifiers (Distraction Buildup & Fatigue)
        if current_state in ["breath_control", "redirect_breath"]:
            # Distraction increases over time in focused states
            progress = min(1.5, current_dwell / max(5, dwell_limit))
            
            # Distraction Pressure: Accumulation of internal stimuli
            distraction_buildup = self.distraction_pressure * progress
            
            for ts in ["pain_discomfort", "pending_tasks"]:
                idx = self.thoughtseeds.index(ts)
                mu[idx] += distraction_buildup
                
            # Cognitive Fatigue: Decay of focus capability
            bf_idx = self.thoughtseeds.index("breath_focus")
            mu[bf_idx] = max(0.1, mu[bf_idx] - (self.fatigue_rate * progress))

        # 2. Set Stochastic Parameters (Ornstein-Uhlenbeck)
        # Theta (Reversion Speed)
        base_theta = DEFAULTS.get('BASE_THETA_NOVICE', 0.2) if self.experience_level == 'novice' else DEFAULTS.get('BASE_THETA_EXPERT', 0.25)
        # Sigma (Volatility)
        base_sigma = DEFAULTS.get('BASE_SIGMA_NOVICE', 0.05) if self.experience_level == 'novice' else DEFAULTS.get('BASE_SIGMA_EXPERT', 0.035)
        
        # 3. Apply OU Update
        for i, ts in enumerate(self.thoughtseeds):
            x_prev = current_activations[i]
            target = mu[i]

            theta = base_theta
            sigma = base_sigma

            # Mind wandering is more volatile and "sticky"
            if current_state == "mind_wandering":
                sigma *= 1.5
                if ts in ["pending_tasks", "pain_discomfort"]:
                    theta *= 0.5

            # Focused states are more stable
            if current_state == "breath_control" and ts == "breath_focus":
                sigma *= 0.5
                theta *= 1.5

            # Calculate update using OU helper
            updated_activations[i] = float(ou_update(x_prev, target, theta, sigma, dt))
            
        from meditation_utils import clip_array
        return clip_array(updated_activations, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

