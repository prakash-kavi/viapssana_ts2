"""
meditation_model.py

Implements core meditation models:
1. RuleBasedLearner: Foundation class providing stochastic thoughtseed dynamics and state transitions.
2. ActInfLearner: Active Inference extension implementing the three-level framework.
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
    NETWORK_PROFILES
)
from meditation_utils import ensure_directories, _save_json_outputs

class RuleBasedLearner:
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
        # Placeholder for previous network activations (initialized in ActInfLearner)
        self.prev_network_acts = None

    def get_target_activations(self, state, meta_awareness):
        """Generate target activations based on state and meta-awareness."""
        targets_dict = ThoughtseedParams.get_target_activations(
            state, meta_awareness, self.experience_level)
        
        target_activations = np.zeros(self.num_thoughtseeds)
        for i, ts in enumerate(self.thoughtseeds):
            target_activations[i] = targets_dict[ts]
        
        # Add noise for biological plausibility
        target_activations += np.random.normal(0, self.noise_level, size=self.num_thoughtseeds)
        return np.clip(target_activations, 0.05, 1.0)

    def get_dwell_time(self, state):
        """Generate random dwell time based on experience level."""
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
        """Calculate meta-awareness based on state and thoughtseed activations."""
        activations_dict = {ts: activations[i] for i, ts in enumerate(self.thoughtseeds)}
        
        return MetacognitionParams.calculate_meta_awareness(
            state=current_state,
            thoughtseed_activations=activations_dict,
            experience_level=self.experience_level
        )

class ActInfLearner(RuleBasedLearner):
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
        
        # Initialize tracking variables
        self.distraction_buildup_rates = []
        self.transition_activations = {state: [] for state in self.states}
        
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
        """
        Compute network activations using physics-based coupled dynamics.
        1. Base Targets (Bottom-up + Top-down)
        2. Coupled Dynamics (FPN->DAN, Mutual Inhibition, VAN Trigger)
        3. Ornstein-Uhlenbeck Process (Smoothing & Stability)
        """
        # Get thoughtseed-to-network contribution matrix
        ts_to_network = self.learned_network_profiles["thoughtseed_contributions"]
        
        # 1. Calculate Base Targets
        target_acts = {net: 0.1 for net in self.networks}
        
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
                fpn_demand = 0.2 + (2.0 * focus_error)
                
                # Experts relax control faster when stable
                efficiency_weight = 0.7 if self.experience_level == 'expert' else 0.3
                target_acts['FPN'] = (1 - efficiency_weight) * target_acts['FPN'] + efficiency_weight * fpn_demand

            # B. FPN Drives DAN + DAN Hysteresis
            # FPN Drive: Top-down recruitment with saturation kinetics
            target_acts['DAN'] += 0.4 * current_fpn * (1.0 - current_dan)
            
            # DAN Hysteresis: Momentum of sustained attention; disrupted by DMN
            current_dmn = self.prev_network_acts.get('DMN', 0)
            hysteresis_strength = 0.2 if self.experience_level == 'expert' else 0.1
            target_acts['DAN'] += hysteresis_strength * current_dan * (1.0 - current_dmn)

        # C. Mutual Inhibition (Task-Positive vs Task-Negative)
        anticorrelation_force = 0.25
        if self.prev_network_acts:
            target_acts['DAN'] -= anticorrelation_force * self.prev_network_acts['DMN'] * target_acts['DAN']
            target_acts['DMN'] -= anticorrelation_force * self.prev_network_acts['DAN'] * target_acts['DMN']
        
        # C. VAN Leaky Integrator (The Detector)
        if self.prev_network_acts:
            current_dmn = self.prev_network_acts['DMN']
            self.dmn_accumulator = 0.9 * self.dmn_accumulator + 0.1 * current_dmn
            
            # Fire if threshold crossed (Refractory period via reset)
            van_trigger_threshold = 0.7
            if self.dmn_accumulator > van_trigger_threshold:
                target_acts['VAN'] += 0.5  # Salience spike
                self.dmn_accumulator = 0.0 # Reset

        # D. FPN Accumulator (Cognitive Fatigue)
        if self.prev_network_acts:
            current_fpn = self.prev_network_acts['FPN']
            
            # Accumulate FPN usage (Leaky Integrator)
            self.fpn_accumulator = 0.98 * self.fpn_accumulator + 0.02 * current_fpn
            
            # Fatigue Threshold: When effort exceeds capacity, focus collapses.
            if self.fpn_accumulator > self.fatigue_threshold:
                target_acts['DAN'] *= 0.6 # Collapse focus
                target_acts['DMN'] += 0.2 # Surge distraction
                self.fpn_accumulator = 0.4 # Reset

        # 3. Ornstein-Uhlenbeck Process (Smoothing & Stability)
        current_acts = {}
        
        if self.prev_network_acts:
            # Theta (θ): Mean reversion rate. Derived from memory_factor.
            theta = 1.0 - self.memory_factor
            sigma = self.noise_level
            
            for net in self.networks:
                x_prev = self.prev_network_acts[net]
                mu = target_acts[net]
                
                # OU Update: x_new = x_prev + theta*(mu - x_prev)*dt + sigma*noise
                noise = np.random.normal(0, 1)
                dx = theta * (mu - x_prev) * dt + sigma * noise
                
                current_acts[net] = x_prev + dx
        else:
            # First step initialization
            current_acts = target_acts
            
        # Normalize 
        for net in self.networks:
            current_acts[net] = np.clip(current_acts[net], 0.05, 0.9)

        # VAN values > 0.85 are neurophysiologically implausible
        max_van = 0.85
        if current_acts['VAN'] > max_van:
            current_acts['VAN'] = max_van
            
        return current_acts
    
    def get_sensory_inference(self, network_acts):
        """
        Infer bottom-up thoughtseed state from network activations.
        Uses the learned Generative Model (A-matrix) to invert observations.
        """
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
                
        return np.clip(inferred, 0.01, 0.99)

    def calculate_vfe(self, current_seeds, prior_seeds, sensory_inference, meta_awareness, vfe_trend=0.0):
        """
        Calculate Variational Free Energy using Negative Log Likelihoods.
        F = (Accuracy * Sensory_Precision) + (Complexity * Prior_Precision)
        """
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
        """
        Update learned network profiles (Eq 3).
        W_ik ← (1-ρ)W_ik + η δ_k(t)z_i(t)
        """
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
        """
        Calculate modulation of thoughtseed targets based on network activity.
        Returns a dictionary of adjustments to be added to the base targets.
        """
        modulations = {ts: 0.0 for ts in self.thoughtseeds}
        
        # DMN enhances pending_tasks and self_reflection, suppresses breath_focus
        dmn_strength = network_acts.get('DMN', 0)
        
        dmn_pending_value = 0.15
        dmn_reflection_value = 0.05 
        dmn_breath_value = 0.2
        
        modulations['pending_tasks'] += dmn_pending_value * dmn_strength
        modulations['self_reflection'] += dmn_reflection_value * dmn_strength
        modulations['breath_focus'] -= dmn_breath_value * dmn_strength

        # VAN enhances pain_discomfort (salience) and self_reflection during meta_awareness
        van_strength = network_acts.get('VAN', 0)
        
        van_pain_value = 0.15
        modulations['pain_discomfort'] += van_pain_value * van_strength
        
        if current_state == "meta_awareness":
            van_reflection_value = 0.2
            modulations['self_reflection'] += van_reflection_value * van_strength

        # DAN enhances breath_focus, suppresses distractions
        dan_strength = network_acts.get('DAN', 0)
        
        dan_breath_value = 0.2
        dan_pending_value = 0.15
        dan_pain_value = 0.1
        
        modulations['breath_focus'] += dan_breath_value * dan_strength
        modulations['pending_tasks'] -= dan_pending_value * dan_strength
        modulations['pain_discomfort'] -= dan_pain_value * dan_strength
        
        # FPN enhances self_reflection and equanimity (metacognition and regulation)
        fpn_strength = network_acts.get('FPN', 0)
        fpn_enhancement = self.fpn_enhancement
        
        fpn_reflection_value = 0.2 if self.experience_level == 'expert' else 0.15
        fpn_equanimity_value = 0.25 if self.experience_level == 'expert' else 0.2
        
        modulations['self_reflection'] += fpn_reflection_value * fpn_strength * fpn_enhancement
        modulations['equanimity'] += fpn_equanimity_value * fpn_strength * fpn_enhancement
                
        return modulations

    def update_thoughtseed_dynamics(self, current_activations, target_activations, current_state, current_dwell, dwell_limit):
        """
        Update thoughtseed activations using Ornstein-Uhlenbeck process.
        """
        dt = 1.0
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
        base_theta = 0.2 if self.experience_level == 'novice' else 0.25
        
        # Sigma (Volatility)
        base_sigma = 0.05 if self.experience_level == 'novice' else 0.035
        
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
            
            # Calculate update
            noise = np.random.normal(0, 1)
            dx = theta * (target - x_prev) * dt + sigma * noise
            
            updated_activations[i] = x_prev + dx
            
        return np.clip(updated_activations, 0.01, 0.99)
    
    def get_meta_awareness(self, current_state, activations):
        """Calculate meta-awareness using the base implementation"""
        return super().get_meta_awareness(current_state, activations)
    
    def get_dwell_time(self, state):
        """Get state-specific dwell time based on experience level."""
        # Use parent method 
        return super().get_dwell_time(state)
    
    def get_transition_probabilities(self, activations, network_acts):
        """
        Calculate transition probabilities based on Expected Free Energy (G).
        Approximated as the similarity between current Thoughtseed State (Affordances)
        and the expected Thoughtseed Profile of each candidate state.
        """
        probs = {}
        
        # 1. Construct Current Observation Vector (Thoughtseeds Only)
        obs_vector = activations
        
        # 2. Calculate Similarity to Each State's Profile
        for state in self.states:
            # Get Expected Thoughtseeds for this state
            ts_targets = ThoughtseedParams.get_target_activations(state, 0.5, self.experience_level)
            state_vector = np.array([ts_targets[ts] for ts in self.thoughtseeds])
            
            # Calculate Similarity (Dot Product)
            # Higher match = Higher probability of transition
            similarity = np.dot(obs_vector, state_vector)
            
            probs[state] = similarity

        # 3. Apply Softmax
        max_score = max(probs.values())
        exp_scores = {k: np.exp((v - max_score) * self.softmax_temperature) for k, v in probs.items()}
        total_exp = sum(exp_scores.values())
        
        return {k: v / total_exp for k, v in exp_scores.items()}

    def train(self, save_outputs=True):
        """
        Train model using active inference.
        1. Calc network activations (Eq 1)
        2. Compute free energy (Eq 2)
        3. Update profiles (Eq 3)
        4. State transitions (Eq 4)
        """
        if save_outputs:
            ensure_directories()
        
        # Initialize training sequence
        state_sequence = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
        current_state_index = 0
        current_state = state_sequence[current_state_index]
        current_dwell = 0
        dwell_limit = self.get_dwell_time(current_state)
        
        # Initialize activations
        activations = np.full(self.num_thoughtseeds, np.random.uniform(0.05, 0.15))
        activations = self.get_target_activations(current_state, 0.6)
        prev_activations = activations.copy()
        
        # Track focused state timing
        time_in_focused_state = 0
        state_transition_patterns = []
        transition_timestamps = []
        
        # Initialize network tracking
        network_acts = self.compute_network_activations(activations, current_state, 0.6)
        prev_network_acts = network_acts.copy()
        
        # Initialize meta-awareness for smoothing
        self.prev_meta_awareness = self.get_meta_awareness(current_state, activations)
        
        # Initialize VFE accumulator for policy switching
        self.vfe_accumulator = 0.0
        
        # Training loop
        for t in range(self.timesteps):
            # --- PASS 1: TOP-DOWN (Action & Prediction) ---
            
            # Calculate instantaneous meta-awareness
            raw_meta_awareness = self.get_meta_awareness(current_state, activations)
            
            # Apply smoothing (EMA)
            smoothing = 0.8 if self.experience_level == 'expert' else 0.6
            
            meta_awareness = smoothing * self.prev_meta_awareness + (1 - smoothing) * raw_meta_awareness
            self.prev_meta_awareness = meta_awareness
            
            if hasattr(self, 'in_transition') and self.in_transition:
                # Continue smoothing the transition over multiple timesteps
                blend_factor = 0.3 * (1.0 + np.random.uniform(-0.1, 0.1))
                
                # Add small random perturbations to transition target
                perturbed_target = self.transition_target.copy()
                perturbed_target += np.random.normal(0, 0.02, size=len(perturbed_target))
                perturbed_target = np.clip(perturbed_target, 0.05, 0.9)
                
                # Apply blending
                activations = (1 - blend_factor) * activations + blend_factor * perturbed_target
                
                self.transition_counter -= 1
                if self.transition_counter <= 0:
                    self.in_transition = False

            # Get target activations (L3 -> L2 Prior)
            target_activations = self.get_target_activations(current_state, meta_awareness)
            
            # Update thoughtseed dynamics (L2 Belief Update)
            activations = self.update_thoughtseed_dynamics(
                activations, 
                target_activations, 
                current_state, 
                current_dwell, 
                dwell_limit
            )
            
            # Handle distraction tracking 
            if current_state in ["breath_control", "redirect_breath"]:
                time_in_focused_state += 1
                progress = min(1.5, current_dwell / max(10, dwell_limit))
                distraction_pressure = 0.4 if self.experience_level == 'novice' else 0.15
                self.distraction_buildup_rates.append(distraction_pressure * progress)
            else:
                time_in_focused_state = 0
                self.distraction_buildup_rates.append(0)
                        
            # --- PASS 2: BOTTOM-UP (Perception & Learning) ---
            
            # Compute network activations (L1 Generative Process)
            network_acts = self.compute_network_activations(activations, current_state, meta_awareness)
            
            # Infer sensory state from networks (L1 -> L2 Sensory Evidence)
            sensory_inference = self.get_sensory_inference(network_acts)
            
            # Calculate VFE (L2 Belief Revision & L3 Monitoring)
            vfe_trend = 0.0
            if len(self.free_energy_history) > 5:
                vfe_trend = np.mean(np.diff(self.free_energy_history[-5:]))

            free_energy, sensory_nll, prior_nll = self.calculate_vfe(
                activations, target_activations, sensory_inference, meta_awareness, vfe_trend
            )
            
            # Update network profiles based on prediction errors
            dummy_errors = {net: sensory_nll * 0.1 for net in self.networks}
            self.update_network_profiles(activations, network_acts, current_state, dummy_errors)
            
            # Record network state and free energy
            self.network_activations_history.append(network_acts.copy())
            self.free_energy_history.append(free_energy)
            self.prediction_error_history.append(sensory_nll)
            self.precision_history.append(0.5 + self.precision_weight * meta_awareness)
            
            # Identify dominant thoughtseed
            dominant_ts = self.thoughtseeds[np.argmax(activations)]
            
            # Track histories
            self.state_history.append(current_state)
            self.activations_history.append(activations.copy())
            self.meta_awareness_history.append(meta_awareness)
            self.dominant_ts_history.append(dominant_ts)
            self.state_history_over_time.append(self.state_indices[current_state])
            
            # --- L3 POLICY SWITCHING (VFE-Based) ---
            transition_happened = False
            
            # Accumulate VFE (Evidence of Policy Failure)
            self.vfe_accumulator = 0.9 * self.vfe_accumulator + 0.1 * free_energy
            
            # Dynamic Threshold based on Experience
            base_threshold = 2.5 if self.experience_level == 'expert' else 3.5
            
            # Check for transition if we've been in the state long enough OR VFE is critical
            critical_vfe = self.vfe_accumulator > (base_threshold * 1.5)
            dwell_expired = current_dwell >= dwell_limit
            
            # Refractory period: Don't force transition via VFE if we just arrived
            min_dwell = STATE_DWELL_TIMES[self.experience_level][current_state][0]
            if dwell_expired or (critical_vfe and current_dwell >= min_dwell):
                # 1. Get Transition Probabilities
                probs = self.get_transition_probabilities(activations, network_acts)
                
                # 2. Force Transition
                if current_state in probs:
                    probs[current_state] = 0.0
                
                # Renormalize
                total_prob = sum(probs.values())
                if total_prob > 0:
                    probs = {k: v / total_prob for k, v in probs.items()}
                else: # pragma: no cover
                    probs = {k: 1.0/(len(probs)-1) for k in probs if k != current_state}
                
                # 3. Sample Next State
                states = list(probs.keys())
                probabilities = list(probs.values())
                next_state = np.random.choice(states, p=probabilities)
                
                # 4. Execute Transition
                transition_happened = True
                self.vfe_accumulator = 0.0
                self.transition_activations[current_state].append(activations.copy())
                self.natural_transition_count += 1
                transition_timestamps.append(t)
                state_transition_patterns.append((
                    current_state, 
                    next_state, 
                    {ts: activations[i] for i, ts in enumerate(self.thoughtseeds)},
                    {net: val for net, val in network_acts.items()},
                    free_energy
                ))
                
                self.transition_counts[current_state][next_state] += 1
                
                # Update state
                if next_state in state_sequence:
                    current_state_index = state_sequence.index(next_state)
                current_state = next_state
                current_dwell = 0
                dwell_limit = self.get_dwell_time(current_state)

                # Calculate new state targets
                new_target = self.get_target_activations(current_state, meta_awareness)

                # Introduce biological variability
                for i in range(len(new_target)):
                    variation = 1.0 + np.random.uniform(-0.05, 0.1)
                    new_target[i] *= variation
                    new_target[i] = max(0.06, new_target[i])

                # Blend current state into new state (Smooth Transition)
                blend_factor = 0.4 * (1.0 + np.random.uniform(-0.1, 0.1))
                activations = (1 - blend_factor) * activations + blend_factor * new_target

                # Add transition markers
                self.in_transition = True
                self.transition_counter = 3 + np.random.randint(0, 2)
                self.transition_target = new_target.copy()

            if not transition_happened:
                current_dwell += 1
            
            # Store for next iteration
            prev_activations = activations.copy()
            prev_network_acts = network_acts.copy()
            self.prev_network_acts = prev_network_acts.copy()
        
        # Save learned weights and network profiles
        if save_outputs:
            ensure_directories()
        
            # Save transition statistics with network data (use centralized aggregates)
            from meditation_utils import compute_state_aggregates

            aggregates = compute_state_aggregates(self)

            transition_stats = {
                'transition_counts': self.transition_counts,
                'transition_thresholds': self.transition_thresholds,
                'natural_transitions': self.natural_transition_count,
                'forced_transitions': self.forced_transition_count,
                'transition_timestamps': transition_timestamps,
                'state_transition_patterns': state_transition_patterns,
                'distraction_buildup_rates': self.distraction_buildup_rates,
                'average_activations_at_transition': aggregates.get('average_activations_at_transition', {}),
                'average_network_activations_by_state': aggregates.get('average_network_activations_by_state', {}),
                'average_free_energy_by_state': aggregates.get('average_free_energy_by_state', {})
            }
            
            # Debug: report network values by state
            logging.info("%s NETWORK VALUES BY STATE:", self.experience_level.upper())
            for state in self.states:
                logging.info("  %s:", state)
                indices = [j for j, s in enumerate(self.state_history) if s == state]
                
                if not indices: # pragma: no cover
                    logging.info("    (No visits to this state)")
                    continue
                    
                state_networks = {
                    net: float(np.mean([
                        self.network_activations_history[j][net]
                        for j in indices
                    ])) for net in self.networks
                }
                for net in self.networks:
                    logging.info("    %s: %.2f", net, state_networks[net])

            # Save the transition statistics
            out_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"transition_stats_{self.experience_level}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(transition_stats, f, indent=2)
            logging.info("Saved transition stats -> %s", os.path.relpath(out_path))

            logging.info("Active Inference training complete for %s.", self.experience_level)
            logging.info("  - Natural transitions: %d, Forced transitions: %d", self.natural_transition_count, self.forced_transition_count)

            # Generate JSON outputs (time series data and parameters)
            _save_json_outputs(self)
