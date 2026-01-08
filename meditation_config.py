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
    """State dwell time configuration"""
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
    """Network profiles for thoughtseeds and states"""
    DMN: float
    VAN: float
    DAN: float
    FPN: float
    
@dataclass
class TransitionThresholds:
    """Thresholds for state transitions"""
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
    """Target activations for each thoughtseed in a particular state"""
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
class ActiveInferenceParameters:
    """Core active inference parameters"""
    precision_weight: float
    complexity_penalty: float
    learning_rate: float
    noise_level: float
    memory_factor: float
    fpn_enhancement: float
    distraction_pressure: float
    fatigue_rate: float
    softmax_temperature: float
    fatigue_threshold: float
    transition_thresholds: TransitionThresholds
    
    @classmethod
    def novice(cls) -> 'ActiveInferenceParameters':
        return cls(
            precision_weight=0.4,
            complexity_penalty=0.4,
            learning_rate=0.01,
            noise_level=0.04,
            memory_factor=0.85, 
            fpn_enhancement=1.0,
            distraction_pressure=1.30,
            fatigue_rate=0.30,
            softmax_temperature=2.5,
            fatigue_threshold=0.50,
            transition_thresholds=TransitionThresholds.novice()
        )
    
    @classmethod
    def expert(cls) -> 'ActiveInferenceParameters':
        return cls(
            precision_weight=0.5,
            complexity_penalty=0.2,
            learning_rate=0.02,
            noise_level=0.03,  
            memory_factor=0.75,  
            fpn_enhancement=1.1, 
            distraction_pressure=0.62,
            fatigue_rate=0.15,         
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
            'distraction_pressure': self.distraction_pressure,
            'fatigue_rate': self.fatigue_rate,
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

# Network profiles - align with meditation neuroscience literature 
# DMN Suppression: Brewer et al., 2022; Hasenkamp & Barsalou, 2012
# DAN/FPN Enhancement: Tang et al., 2015; Lutz et al., 2008; Malinowski (2013)
# Salience Detection and VAN: Seeley et al., 2007; Menon & Uddin, 2010; Kirk et al., 2016; Farb et al., 2007
# DMN-DAN Anticorrelation: Fox et al., 2005; Spreng et al., 2013; Mooneyham et al., 2017; Josipovic et al. 2012
# Meta-awareness and Mind Wandering: Lutz et al., 2015; Christoff et al., 2009; Fox et al., 2015
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
    """Thoughtseed target activation parameters derived from meditation literature."""
    
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
    """Meta-awareness parameters assumptions based on empirical studies of meditation."""
    
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

class ActiveInferenceConfig:
    """
    Centralized configuration for active inference parameters.
    Access parameters with ActiveInferenceConfig.get_params(experience_level)
    """
    @staticmethod
    def get_params(experience_level):
        """Get all active inference parameters for the specified experience level"""
        if experience_level == 'expert':
            return ActiveInferenceParameters.expert().as_dict()
        else:
            return ActiveInferenceParameters.novice().as_dict()
