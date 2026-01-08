"""
meditation_config.py

This file contains configuration data for the Vipassana Entropy meditation simulation,
using dataclasses for improved type safety and maintainability.
"""
from dataclasses import dataclass, field
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
    
    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary for compatibility with existing code"""
        return {
            "breath_focus": self.breath_focus,
            "equanimity": self.equanimity,
            "pain_discomfort": self.pain_discomfort,
            "pending_tasks": self.pending_tasks,
            "self_reflection": self.self_reflection
        }

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
            softmax_temperature=2.5,
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
            softmax_temperature=2.0,   
            fatigue_threshold=0.75,
            transition_thresholds=TransitionThresholds.expert()
        )
    
    def as_dict(self) -> Dict[str, Union[float, Dict]]:
        """Convert to dictionary for compatibility with existing code"""
        return {
            'precision_weight': self.precision_weight,
            'complexity_penalty': self.complexity_penalty,
            'learning_rate': self.learning_rate,
            'noise_level': self.noise_level,
            'memory_factor': self.memory_factor,
            'fpn_enhancement': self.fpn_enhancement,
            'base_theta': self.base_theta,
            'base_sigma': self.base_sigma,
            'fpn_reflection_value': self.fpn_reflection_value,
            'fpn_equanimity_value': self.fpn_equanimity_value,
            'distraction_pressure': self.distraction_pressure,
            'fatigue_rate': self.fatigue_rate,
            'smoothing': self.smoothing,
            'blend_factor_transition': self.blend_factor_transition,
            'blend_factor_state': self.blend_factor_state,
            'blend_variation': self.blend_variation,
            'transition_perturb_std': self.transition_perturb_std,
            'transition_variation_low': self.transition_variation_low,
            'transition_variation_high': self.transition_variation_high,
            'vfe_accum_decay': self.vfe_accum_decay,
            'vfe_accum_alpha': self.vfe_accum_alpha,
            'fpn_accum_decay': self.fpn_accum_decay,
            'fpn_accum_inc': self.fpn_accum_inc,
            'fatigue_reset': self.fatigue_reset,
            'fpn_collapse_dan_mult': self.fpn_collapse_dan_mult,
            'fpn_collapse_dmn_inc': self.fpn_collapse_dmn_inc,
            'fpn_base_demand': self.fpn_base_demand,
            'fpn_focus_mult': self.fpn_focus_mult,
            'softmax_temperature': self.softmax_temperature,
            'fatigue_threshold': self.fatigue_threshold,
            'transition_thresholds': {
                'mind_wandering': self.transition_thresholds.mind_wandering,
                'dmn_dan_ratio': self.transition_thresholds.dmn_dan_ratio,
                'meta_awareness': self.transition_thresholds.meta_awareness,
                'return_focus': self.transition_thresholds.return_focus
            }
        }

# Create the base configurations as module-level constants
STATE_DWELL_TIMES = {
    'novice': DwellTimeConfig.novice().__dict__,
    'expert': DwellTimeConfig.expert().__dict__
}

# Project-wide numeric defaults to avoid magic numbers
DEFAULTS = {
    'TARGET_CLIP_MIN': 0.05,
    'TARGET_CLIP_MAX': 1.0,
    'ACTIVATION_CLIP_MIN': 0.01,
    'ACTIVATION_CLIP_MAX': 0.99,
    'NETWORK_CLIP_MIN': 0.05,
    'NETWORK_CLIP_MAX': 0.9,
    'VAN_TRIGGER': 0.7,
    'VAN_MAX': 0.85,
    'DEFAULT_DT': 1.0,
    
    'ANTICORRELATION_FORCE': 0.25,
    'EFFICIENCY_WEIGHT_EXPERT': 0.7,
    'EFFICIENCY_WEIGHT_NOVICE': 0.3,
    'MIN_HISTORY_FOR_LEARNING': 10
}

# Additional tunable defaults for dynamics and transitions
DEFAULTS.update({
    'NETWORK_BASE': 0.1,
    'FPN_TO_DAN_GAIN': 0.4,
    'HYSTERESIS_EXPERT': 0.2,
    'HYSTERESIS_NOVICE': 0.1,
    'VAN_SPIKE': 0.5,
    
    'TRANSITION_COUNTER_BASE': 3,
    'TRANSITION_COUNTER_RAND': 2,
    
})

# Additional surface/modulation defaults (moved from hard-coded locations)
DEFAULTS.update({
    'DMN_PENDING_VALUE': 0.15,
    'DMN_REFLECTION_VALUE': 0.05,
    'DMN_BREATH_VALUE': 0.2,
    'VAN_PAIN_VALUE': 0.15,
    'VAN_REFLECTION_VALUE': 0.2,
    'DAN_BREATH_VALUE': 0.2,
    'DAN_PENDING_VALUE': 0.15,
    'DAN_PAIN_VALUE': 0.1,
 
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
        "breath_control": StateTargetActivations(
            breath_focus=0.7,
            equanimity=0.3,
            pain_discomfort=0.15,
            pending_tasks=0.1,
            self_reflection=0.2
        ).as_dict(),
        "mind_wandering": StateTargetActivations(
            breath_focus=0.1,
            equanimity=0.1,
            pain_discomfort=0.6,
            pending_tasks=0.7,
            self_reflection=0.1
        ).as_dict(),
        "meta_awareness": StateTargetActivations(
            breath_focus=0.2,
            equanimity=0.3,
            pain_discomfort=0.15,
            pending_tasks=0.15,
            self_reflection=0.8
        ).as_dict(),
        "redirect_breath": StateTargetActivations(
            breath_focus=0.6,
            equanimity=0.7,
            pain_discomfort=0.2,
            pending_tasks=0.1,
            self_reflection=0.4
        ).as_dict()
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
            "pain_discomfort": 0.4,   # Positive target: MW expects pain
            "pending_tasks": 0.4,     # Positive target: MW expects tasks
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
        """
        Calculate meta-awareness based on state, thoughtseed activations, and experience.
        """
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

def get_actinf_params_dict(experience_level: str) -> Dict[str, Union[float, Dict]]:
    """Return a params dict for the requested experience level using `ActInfParams` dataclass."""
    return ActInfParams.expert().as_dict() if experience_level == 'expert' else ActInfParams.novice().as_dict()
