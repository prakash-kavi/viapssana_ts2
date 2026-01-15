"""Configuration for the Vipassana Entropy meditation simulation.

Note: external JSON-based config loading (e.g. `config/config.json`) is
possible for experiment workflows but is not enabled in this file.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# Core thoughtseed and state definitions
THOUGHTSEEDS = ['breath_focus', 'pain_discomfort', 'pending_tasks', 'self_reflection', 'equanimity']
STATES = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']

@dataclass
class DwellTimeConfig:
    breath_control: Tuple[int, int]
    mind_wandering: Tuple[int, int]
    meta_awareness: Tuple[int, int]
    redirect_breath: Tuple[int, int]
    
    @classmethod
    def novice(cls) -> 'DwellTimeConfig':
        return cls(
            breath_control=(5, 15),
            mind_wandering=(22, 42),
            meta_awareness=(2, 7),
            redirect_breath=(2, 5)
        )
    
    @classmethod
    def expert(cls) -> 'DwellTimeConfig':
        return cls(
            breath_control=(15, 25),
            mind_wandering=(8, 12),
            meta_awareness=(2, 4),
            redirect_breath=(2, 4)
        )

@dataclass
class NetworkProfile:
    DMN: float
    VAN: float
    DAN: float
    FPN: float
    
@dataclass
class TransitionThresholds:
    mind_wandering: float  # Distraction level threshold
    dmn_dan_ratio: float   # DMN/DAN ratio threshold
    meta_awareness: float  # Self-reflection threshold for meta-awareness
    return_focus: float    # Threshold to return to focused states
    
    @classmethod
    def novice(cls) -> 'TransitionThresholds':
        return cls(
            mind_wandering=0.6,
            dmn_dan_ratio=0.5,
            meta_awareness=0.4,
            return_focus=0.3
        )
    
    @classmethod
    def expert(cls) -> 'TransitionThresholds':
        return cls(
            mind_wandering=0.65,
            dmn_dan_ratio=0.6,
            meta_awareness=0.3,
            return_focus=0.35
        )

@dataclass
class StateTargetActivations:
    breath_focus: float
    equanimity: float
    pain_discomfort: float
    pending_tasks: float
    self_reflection: float
    
    # Instances can be converted to dicts via `dataclasses.asdict` where needed.

@dataclass
class ActInfParams:
    precision_weight: float
    complexity_penalty: float
    learning_rate: float
    noise_level: float
    memory_factor: float
    fpn_enhancement: float
    fpn_reflection_value: float
    fpn_equanimity_value: float
    distraction_pressure: float
    fatigue_rate: float
    fpn_accum_decay: float
    fpn_accum_inc: float
    fatigue_reset: float
    # FPN collapse / base demand tunables
    fpn_collapse_dan_mult: float
    fpn_collapse_dmn_inc: float
    fpn_base_demand: float
    fpn_focus_mult: float
    # Network/dynamics tunables (migrated from DEFAULTS)
    network_base: float
    fpn_to_dan_gain: float
    hysteresis_strength: float
    anticorrelation_force: float
    van_spike: float
    # Surface / modulation defaults (DMN/VAN/DAN effects)
    dmn_pending_value: float
    dmn_reflection_value: float
    dmn_breath_value: float
    van_pain_value: float
    van_reflection_value: float
    dan_breath_value: float
    dan_pending_value: float
    dan_pain_value: float
    # Efficiency weight (expert vs novice differences)
    efficiency_weight: float
    # Per-agent smoothing/blending and transition noise
    smoothing: float
    blend_factor_transition: float
    blend_factor_state: float
    blend_variation: float
    transition_perturb_std: float
    transition_variation_low: float
    transition_variation_high: float
    # VFE accumulator dynamics
    vfe_accum_decay: float
    vfe_accum_alpha: float
    base_theta: float
    base_sigma: float
    softmax_temperature: float
    fatigue_threshold: float
    transition_thresholds: TransitionThresholds
    
    @classmethod
    def novice(cls) -> 'ActInfParams':
        return cls(
            precision_weight=0.4,
            complexity_penalty=0.4,
            learning_rate=0.01,
            noise_level=0.04,
            memory_factor=0.85, 
            fpn_enhancement=1.0,
            fpn_reflection_value=0.15,
            fpn_equanimity_value=0.2,
            distraction_pressure=1.30,
            fatigue_rate=0.30,
            smoothing=0.6,
            blend_factor_transition=0.3,
            blend_factor_state=0.4,
            blend_variation=0.1,
            transition_perturb_std=0.02,
            transition_variation_low=-0.05,
            transition_variation_high=0.1,
            vfe_accum_decay=0.9,
            vfe_accum_alpha=0.1,
            base_theta=0.2,
            base_sigma=0.05,
            fpn_accum_decay=0.98,
            fpn_accum_inc=0.02,
            fatigue_reset=0.4,
            fpn_collapse_dan_mult=0.6,
            fpn_collapse_dmn_inc=0.2,
            fpn_base_demand=0.2,
            fpn_focus_mult=2.0,
            # network/dynamics defaults (may be overridden by config/*.json)
            network_base=0.1,
            fpn_to_dan_gain=0.4,
            hysteresis_strength=0.1,
            anticorrelation_force=0.25,
            van_spike=0.5,
            # surface/modulation defaults
            dmn_pending_value=0.15,
            dmn_reflection_value=0.05,
            dmn_breath_value=0.2,
            van_pain_value=0.15,
            van_reflection_value=0.2,
            dan_breath_value=0.2,
            dan_pending_value=0.15,
            dan_pain_value=0.1,
            softmax_temperature=2.5,
            efficiency_weight=0.3,
            fatigue_threshold=0.50,
            transition_thresholds=TransitionThresholds.novice()
        )
    
    @classmethod
    def expert(cls) -> 'ActInfParams':
        return cls(
            precision_weight=0.5,
            complexity_penalty=0.2,
            learning_rate=0.02,
            noise_level=0.03,  
            memory_factor=0.75,  
            fpn_enhancement=1.1, 
            fpn_reflection_value=0.2,
            fpn_equanimity_value=0.25,
            distraction_pressure=0.62,
            fatigue_rate=0.15,         
            smoothing=0.8,
            blend_factor_transition=0.3,
            blend_factor_state=0.4,
            blend_variation=0.1,
            transition_perturb_std=0.02,
            transition_variation_low=-0.05,
            transition_variation_high=0.1,
            vfe_accum_decay=0.9,
            vfe_accum_alpha=0.1,
            base_theta=0.25,
            base_sigma=0.035,
            fpn_accum_decay=0.98,
            fpn_accum_inc=0.02,
            fatigue_reset=0.4,
            fpn_collapse_dan_mult=0.6,
            fpn_collapse_dmn_inc=0.2,
            fpn_base_demand=0.2,
            fpn_focus_mult=2.0,
            # network/dynamics defaults (may be overridden by config/*.json)
            network_base=0.1,
            fpn_to_dan_gain=0.4,
            hysteresis_strength=0.2,
            anticorrelation_force=0.25,
            van_spike=0.5,
            # surface/modulation defaults
            dmn_pending_value=0.15,
            dmn_reflection_value=0.05,
            dmn_breath_value=0.2,
            van_pain_value=0.15,
            van_reflection_value=0.2,
            dan_breath_value=0.2,
            dan_pending_value=0.15,
            dan_pain_value=0.1,
            softmax_temperature=2.0,   
            efficiency_weight=0.7,
            fatigue_threshold=0.75,
            transition_thresholds=TransitionThresholds.expert()
        )

# Create the base configurations as module-level constants
STATE_DWELL_TIMES = {
    # Per-experience-level dwell time ranges (min, max) for each state
    'novice': DwellTimeConfig.novice().__dict__,
    'expert': DwellTimeConfig.expert().__dict__
}

# Project-wide numeric defaults to avoid magic numbers
DEFAULTS = {
    # Numeric clamps and thresholds used across the simulation
    'TARGET_CLIP_MIN': 0.05,  # lower bound for thoughtseed target activations
    'TARGET_CLIP_MAX': 1.0,   # upper bound for thoughtseed target activations
    'ACTIVATION_CLIP_MIN': 0.01,
    'ACTIVATION_CLIP_MAX': 0.99,
    'NETWORK_CLIP_MIN': 0.05,  # network activation lower bound
    'NETWORK_CLIP_MAX': 0.9,   # network activation upper bound
    'VAN_TRIGGER': 0.7,        # VAN accumulator threshold for salience spike
    'VAN_MAX': 0.85,           # physiological cap for VAN
    'DEFAULT_DT': 1.0,
    'MIN_HISTORY_FOR_LEARNING': 10
}

# Additional tunable defaults for dynamics and transitions
DEFAULTS.update({
    'TRANSITION_COUNTER_BASE': 3,
    'TRANSITION_COUNTER_RAND': 2,
})

# Network profiles for thoughtseeds and states
NETWORK_PROFILES = {
    "thoughtseed_contributions": {
        "breath_focus": NetworkProfile(DMN=0.2, VAN=0.3, DAN=0.65, FPN=0.6).__dict__,
        "pain_discomfort": NetworkProfile(DMN=0.5, VAN=0.7, DAN=0.3, FPN=0.4).__dict__,
        "pending_tasks": NetworkProfile(DMN=0.8, VAN=0.5, DAN=0.2, FPN=0.4).__dict__,
        "self_reflection": NetworkProfile(DMN=0.6, VAN=0.4, DAN=0.3, FPN=0.8).__dict__,
        "equanimity": NetworkProfile(DMN=0.3, VAN=0.3, DAN=0.5, FPN=0.9).__dict__
    },
    
    # State profiles differentiated by experience level
    # Expected network activations per high-level state and experience level
    "state_expected_profiles": {
        # BREATH CONTROL: Experts have lower DMN, higher DAN/FPN
        "breath_control": {
            "novice": NetworkProfile(DMN=0.35, VAN=0.4, DAN=0.7, FPN=0.5).__dict__,
            "expert": NetworkProfile(DMN=0.24, VAN=0.42, DAN=0.68, FPN=0.65).__dict__
        },
        
        # MIND WANDERING: Experts have much lower DMN, higher FPN control
        "mind_wandering": {
            "novice": NetworkProfile(DMN=0.85, VAN=0.45, DAN=0.2, FPN=0.35).__dict__,
            "expert": NetworkProfile(DMN=0.55, VAN=0.55, DAN=0.35, FPN=0.50).__dict__
        },
        
        # META-AWARENESS: Experts have higher VAN (detection) and FPN (control)
        "meta_awareness": {
            "novice": NetworkProfile(DMN=0.35, VAN=0.7, DAN=0.5, FPN=0.45).__dict__,
            "expert": NetworkProfile(DMN=0.32, VAN=0.85, DAN=0.48, FPN=0.55).__dict__
        },
        
        # REDIRECT BREATH: Experts have lower DMN, higher DAN/FPN (control)
        "redirect_breath": {
            "novice": NetworkProfile(DMN=0.3, VAN=0.45, DAN=0.65, FPN=0.55).__dict__,
            "expert": NetworkProfile(DMN=0.18, VAN=0.55, DAN=0.68, FPN=0.65).__dict__
        }
    }
}

@dataclass
class ThoughtseedParams:
    
    # Base target activation patterns for each thoughtseed in each state
    BASE_ACTIVATIONS = {
        "breath_control": asdict(StateTargetActivations(
            breath_focus=0.7,
            equanimity=0.3,
            pain_discomfort=0.15,
            pending_tasks=0.1,
            self_reflection=0.2
        )),
        "mind_wandering": asdict(StateTargetActivations(
            breath_focus=0.1,
            equanimity=0.1,
            pain_discomfort=0.6,
            pending_tasks=0.7,
            self_reflection=0.1
        )),
        "meta_awareness": asdict(StateTargetActivations(
            breath_focus=0.2,
            equanimity=0.3,
            pain_discomfort=0.15,
            pending_tasks=0.15,
            self_reflection=0.8
        )),
        "redirect_breath": asdict(StateTargetActivations(
            breath_focus=0.6,
            equanimity=0.7,
            pain_discomfort=0.2,
            pending_tasks=0.1,
            self_reflection=0.4
        ))
    }
    
    # How meta-awareness modulates each thoughtseed in each state
    META_AWARENESS_MODULATORS = {
        "breath_control": {
            "breath_focus": 0.1,
            "equanimity": 0.25,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.1
        },
        "mind_wandering": {
            "breath_focus": 0.0,
            "equanimity": -0.05,
            "pain_discomfort": -0.1,
            "pending_tasks": -0.1,
            "self_reflection": 0.3
        },
        "meta_awareness": {
            "breath_focus": 0.1,
            "equanimity": 0.1,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.1
        },
        "redirect_breath": {
            "breath_focus": 0.2,
            "equanimity": 0.25,
            "pain_discomfort": -0.1,
            "pending_tasks": 0.0,
            "self_reflection": 0.1
        }
    }
    
    # Experience-specific adjustments (values to add for experts)
    EXPERT_ADJUSTMENTS = {
        "breath_control": {
            "breath_focus": 0.1,
            "equanimity": 0.2,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.0
        },
        "mind_wandering": {
            "breath_focus": 0.0,
            "equanimity": 0.05,
            "pain_discomfort": 0.4,
            "pending_tasks": 0.4,
            "self_reflection": 0.0
        },
        "meta_awareness": {
            "breath_focus": 0.0,
            "equanimity": 0.1,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.1
        },
        "redirect_breath": {
            "breath_focus": 0.0,
            "equanimity": 0.2,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.0
        }
    }
    
    @staticmethod
    def get_target_activations(state, meta_awareness, experience_level='novice'):
        """Get target activation values for each thoughtseed in the specified state."""
        # Start with base activations for this state
        activations = ThoughtseedParams.BASE_ACTIVATIONS[state].copy()
        
        # Apply meta-awareness modulation
        for ts in activations:
            modulator = ThoughtseedParams.META_AWARENESS_MODULATORS[state][ts]
            activations[ts] += modulator * meta_awareness
        
        # Apply expert adjustments if applicable
        if experience_level == 'expert':
            for ts in activations:
                activations[ts] += ThoughtseedParams.EXPERT_ADJUSTMENTS[state].get(ts, 0)
        
        return activations

@dataclass
class MetacognitionParams:
    
    # Base meta-awareness levels for each state
    BASE_AWARENESS = {
        "breath_control": 0.4,
        "mind_wandering": 0.2,
        "meta_awareness": 0.6,
        "redirect_breath": 0.5
    }
    
    # How thoughtseeds influence meta-awareness
    THOUGHTSEED_INFLUENCES = {
        "self_reflection": 0.1,  # Self-reflection strongly enhances meta-awareness
        "equanimity": 0.1        # Equanimity provides a stronger regulation boost for experts
    }
    
    @staticmethod
    def calculate_meta_awareness(state, thoughtseed_activations, experience_level='novice'):
        """Compute meta-awareness from state and thoughtseed activations."""
        # Get base awareness for this state
        base_awareness = MetacognitionParams.BASE_AWARENESS[state]
        
        # Calculate thoughtseed influence
        awareness_boost = 0
        for ts, influence in MetacognitionParams.THOUGHTSEED_INFLUENCES.items():
            if ts in thoughtseed_activations:
                awareness_boost += thoughtseed_activations[ts] * influence
        
        # Calculate total (without noise)
        meta_awareness = base_awareness + awareness_boost
        
        return meta_awareness