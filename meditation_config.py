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
class EffectMagnitude:
    """Standardized effect magnitudes for model adjustments"""
    NONE: float = 0.0
    WEAK: float = 0.1
    MODERATE: float = 0.2
    STRONG: float = 0.3

@dataclass
class ContextualEffect:
    """Effect that varies by meditation state and practitioner experience"""
    base_value: float
    state_modifiers: Dict[str, float] = field(default_factory=dict)
    experience_modifiers: Dict[str, float] = field(default_factory=dict)
    
    def get_value(self, state, experience_level):
        """Calculate contextual value based on state and experience"""
        value = self.base_value
        value += self.state_modifiers.get(state, 0.0)
        value += self.experience_modifiers.get(experience_level, 0.0)
        return value

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
            mind_wandering=(15, 30),
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
class ThoughtseedAgent:
    """Configuration for a single thoughtseed agent"""
    id: int
    category: str
    intentional_weights: Dict[str, float]
    decay_rate: float
    recovery_rate: float

@dataclass
class NetworkProfile:
    """Network profiles for thoughtseeds and states"""
    DMN: float
    VAN: float
    DAN: float
    FPN: float
    
@dataclass
class NonLinearDynamicsConfig:
    """Configuration for non-linear network dynamics"""
    # Compression thresholds
    high_compression_threshold: float = 0.8
    low_compression_threshold: float = 0.2
    
    # Compression factors
    high_compression_factor: float = EffectMagnitude.MODERATE
    low_compression_factor: float = EffectMagnitude.MODERATE
    
    # DMN-DAN anticorrelation boundary factors
    anticorrelation_boundary_threshold: float = 0.3
    
    # Extreme value prevention
    max_dmn_mind_wandering: float = 0.85
    
    # Rapid change detection
    rapid_change_threshold: float = 0.15
    rapid_change_smoothing: float = 0.7

@dataclass
class NetworkModulationConfig:
    """Centralized configuration for all network modulation effects"""
    
    # Dictionary fields need to use field(default_factory=dict)
    dmn_boost: Dict[str, ContextualEffect] = field(default_factory=dict)
    dmn_suppression: Dict[str, ContextualEffect] = field(default_factory=dict)
    dan_boost: Dict[str, ContextualEffect] = field(default_factory=dict)
    dan_suppression: Dict[str, ContextualEffect] = field(default_factory=dict)
    van_boost: Dict[str, ContextualEffect] = field(default_factory=dict)
    fpn_boost: Dict[str, ContextualEffect] = field(default_factory=dict)
    
    # These fields need to use field(default=None) instead of =None
    dmn_dan_anticorrelation: Optional[ContextualEffect] = field(default=None)
    dan_dmn_anticorrelation: Optional[ContextualEffect] = field(default=None)
    memory_factor: Optional[ContextualEffect] = field(default=None)
    
    # Non-linear dynamics configuration
    non_linear_dynamics: NonLinearDynamicsConfig = field(
        default_factory=NonLinearDynamicsConfig
    )
    
    @classmethod
    def novice(cls) -> 'NetworkModulationConfig':
        """Create configuration for novice practitioners"""
        config = cls(
            # Network interactions
            dmn_dan_anticorrelation=ContextualEffect(
                base_value=EffectMagnitude.WEAK,
                state_modifiers={"mind_wandering": EffectMagnitude.WEAK}
            ),
            dan_dmn_anticorrelation=ContextualEffect(
                base_value=EffectMagnitude.WEAK
            ),
            # Memory factor
            memory_factor=ContextualEffect(
                base_value=0.7
            )
        )
        
        # DMN boosts
        config.dmn_boost["mind_wandering"] = ContextualEffect(
            base_value=EffectMagnitude.WEAK,  # Lower boost for mind wandering (0.1)
            state_modifiers={}
        )
        config.dmn_suppression["breath_control"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Strong suppression (0.3)
        )
        config.dmn_suppression["redirect_breath"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate suppression (0.2)
        )
        
        config.dmn_suppression["meta_awareness"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Suppression for meta_awareness (0.2)
        )
        
        # DAN boosts and suppressions
        config.dan_boost["breath_control"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate boost (0.2)
        )
        config.dan_boost["redirect_breath"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate boost (0.2)
        )
        config.dan_suppression["mind_wandering"] = ContextualEffect(
            base_value=EffectMagnitude.WEAK  # Weak suppression (0.1)
        )
        
        # VAN boosts
        config.van_boost["meta_awareness"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate boost (0.2)
        )
        config.van_boost["pain_detection"] = ContextualEffect(
            base_value=EffectMagnitude.WEAK  # Weak boost (0.1)
        )
        
        # FPN boosts
        config.fpn_boost["meta_awareness"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate boost (0.2)
        )
        config.fpn_boost["redirect_breath"] = ContextualEffect(
            base_value=EffectMagnitude.WEAK  # Weak boost (0.1)
        )
        
        return config
    
    @classmethod
    def expert(cls) -> 'NetworkModulationConfig':
        """Create configuration for expert practitioners"""
        config = cls(
            # Network interactions - weaker for experts to reduce volatility
            dmn_dan_anticorrelation=ContextualEffect(
                base_value=EffectMagnitude.WEAK,
                state_modifiers={"mind_wandering": EffectMagnitude.WEAK}
            ),
            dan_dmn_anticorrelation=ContextualEffect(
                base_value=EffectMagnitude.STRONG,
                state_modifiers={"mind_wandering": EffectMagnitude.STRONG}
            ),
            # Memory factor - higher for experts (more temporal stability)
            memory_factor=ContextualEffect(
                base_value=0.85
            )
        )
        
        # DMN boosts
        config.dmn_boost["mind_wandering"] = ContextualEffect(
            base_value=EffectMagnitude.WEAK,  # Keep DMN engagement present but subdued (0.1)
            state_modifiers={}
        )
        config.dmn_suppression["breath_control"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate suppression (0.2)
        )
        config.dmn_suppression["redirect_breath"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate suppression (0.2)
        )
        
        # DAN boosts and suppressions
        config.dan_boost["breath_control"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate boost (0.2)
        )
        config.dan_boost["redirect_breath"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate boost (0.2)
        )
        config.dan_suppression["mind_wandering"] = ContextualEffect(
            base_value=EffectMagnitude.STRONG  # Strong suppression (0.3) 
        )
        
        # VAN boosts
        config.van_boost["meta_awareness"] = ContextualEffect(
            base_value=EffectMagnitude.STRONG  # Strong boost (0.3)
        )
        config.van_boost["pain_detection"] = ContextualEffect(
            base_value=EffectMagnitude.WEAK  # Weak boost (0.1)
        )
        
        # FPN boosts
        config.fpn_boost["meta_awareness"] = ContextualEffect(
            base_value=EffectMagnitude.STRONG  # Strong boost (0.3)
        )
        config.fpn_boost["redirect_breath"] = ContextualEffect(
            base_value=EffectMagnitude.MODERATE  # Moderate boost (0.2)
        )
        
        return config

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
            mind_wandering=0.75,
            dmn_dan_ratio=0.65,
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
    transition_thresholds: TransitionThresholds
    network_modulation: NetworkModulationConfig
    
    @classmethod
    def novice(cls) -> 'ActiveInferenceParameters':
        return cls(
            precision_weight=0.4,
            complexity_penalty=0.4,
            learning_rate=0.01,
            noise_level=0.06,  
            memory_factor=0.7,
            fpn_enhancement=1.0,
            transition_thresholds=TransitionThresholds.novice(),
            network_modulation=NetworkModulationConfig.novice()
        )
    
    @classmethod
    def expert(cls) -> 'ActiveInferenceParameters':
        return cls(
            precision_weight=0.5,  
            complexity_penalty=0.2,
            learning_rate=0.02,
            noise_level=0.03,  
            memory_factor=0.85,  
            fpn_enhancement=1.2,
            transition_thresholds=TransitionThresholds.expert(),
            network_modulation=NetworkModulationConfig.expert()
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
            'transition_thresholds': {
                'mind_wandering': self.transition_thresholds.mind_wandering,
                'dmn_dan_ratio': self.transition_thresholds.dmn_dan_ratio,
                'meta_awareness': self.transition_thresholds.meta_awareness,
                'return_focus': self.transition_thresholds.return_focus
            },
            'network_modulation': self.network_modulation.__dict__
        }

# Create the base configurations as module-level constants
STATE_DWELL_TIMES = {
    'novice': DwellTimeConfig.novice().__dict__,
    'expert': DwellTimeConfig.expert().__dict__
}

# Thoughtseed agents
THOUGHTSEED_AGENTS = {
    "breath_focus": ThoughtseedAgent(
        id=0, 
        category="focus", 
        intentional_weights={"novice": 0.8, "expert": 0.95}, 
        decay_rate=0.005, 
        recovery_rate=0.06
    ).__dict__,
    "pain_discomfort": ThoughtseedAgent(
        id=1, 
        category="distraction", 
        intentional_weights={"novice": 0.4, "expert": 0.6}, 
        decay_rate=0.003, 
        recovery_rate=0.05
    ).__dict__,
    "pending_tasks": ThoughtseedAgent(
        id=2, 
        category="distraction", 
        intentional_weights={"novice": 0.3, "expert": 0.5}, 
        decay_rate=0.002, 
        recovery_rate=0.03
    ).__dict__,
    "self_reflection": ThoughtseedAgent(
        id=3, 
        category="metacognition", 
        intentional_weights={"novice": 0.5, "expert": 0.8}, 
        decay_rate=0.004, 
        recovery_rate=0.04
    ).__dict__,
    "equanimity": ThoughtseedAgent(
        id=4, 
        category="regulation", 
        intentional_weights={"novice": 0.5, "expert": 0.9}, 
        decay_rate=0.001, 
        recovery_rate=0.02
    ).__dict__
}

# Network profiles - align with meditation neuroscience literature 
# DMN Suppression: Brewer et al., 2022; Hasenkamp & Barsalou, 2012
# DAN/FPN Enhancement: Tang et al., 2015; Lutz et al., 2008; Malinowski (2013)
# Salience Detection and VAN: Seeley et al., 2007; Menon & Uddin, 2010; Kirk et al., 2016; Farb et al., 2007
# DMN-DAN Anticorrelation: Fox et al., 2005; Spreng et al., 2013; Mooneyham et al., 2017; Josipovic et al. 2012
# Meta-awreness and Mind Wandering: Lutz et al., 2015; Christoff et al., 2009; Fox et al., 2015
NETWORK_PROFILES = {
    "thoughtseed_contributions": {
        "breath_focus": NetworkProfile(DMN=0.2, VAN=0.3, DAN=0.8, FPN=0.6).__dict__,
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
            "expert": NetworkProfile(DMN=0.24, VAN=0.42, DAN=0.82, FPN=0.75).__dict__
        },
        
        # MIND WANDERING: Experts have much lower DMN, higher FPN control
        "mind_wandering": {
            "novice": NetworkProfile(DMN=0.85, VAN=0.45, DAN=0.2, FPN=0.35).__dict__,
            "expert": NetworkProfile(DMN=0.55, VAN=0.55, DAN=0.4, FPN=0.6).__dict__
        },
        
        # META-AWARENESS: Experts have higher VAN (detection) and FPN (control)
        "meta_awareness": {
            "novice": NetworkProfile(DMN=0.35, VAN=0.7, DAN=0.5, FPN=0.45).__dict__,
            "expert": NetworkProfile(DMN=0.32, VAN=0.85, DAN=0.58, FPN=0.65).__dict__
        },
        
        # REDIRECT BREATH: Experts have lower DMN, higher DAN/FPN (control)
        "redirect_breath": {
            "novice": NetworkProfile(DMN=0.3, VAN=0.45, DAN=0.65, FPN=0.55).__dict__,
            "expert": NetworkProfile(DMN=0.18, VAN=0.55, DAN=0.82, FPN=0.75).__dict__
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
            "pain_discomfort": -0.1,  
            "pending_tasks": -0.3,   
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
        "breath_control": 0.5,
        "mind_wandering": 0.25,
        "meta_awareness": 0.7,
        "redirect_breath": 0.6
    }
    
    # How thoughtseeds influence meta-awareness
    THOUGHTSEED_INFLUENCES = {
        "self_reflection": 0.2,  # Self-reflection strongly enhances meta-awareness
        "equanimity": 0.2        # Equanimity provides a stronger regulation boost for experts
    }
    
    # Experience level adjustments
    EXPERIENCE_BOOST= {
        "novice": 0.0,
        "expert": 0.2
    }
    
    # Expert efficiency adjustments (multiplication factors)
    EXPERT_EFFICIENCY = {
        "meta_awareness": 0.8,   # Lower explicit but more efficient
        "other_states": 0.9      # Better background meta-awareness
    }
    
    @staticmethod
    def get_base_awareness(state):
        """Return base meta-awareness level for the given state"""
        return MetacognitionParams.BASE_AWARENESS[state]
    
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
        
        # Add experience boost
        experience_boost = MetacognitionParams.EXPERIENCE_BOOST[experience_level]
        
        # Calculate total (without noise)
        meta_awareness = base_awareness + awareness_boost + experience_boost
        
        # Apply expert efficiency adjustments if applicable
        if experience_level == 'expert':
            if state == "meta_awareness":
                # Experts show lower explicit meta-awareness values but more efficient processing
                meta_awareness *= MetacognitionParams.EXPERT_EFFICIENCY["meta_awareness"]
            else:
                # Experts maintain better background meta-awareness in other states
                meta_awareness = max(0.3, meta_awareness * MetacognitionParams.EXPERT_EFFICIENCY["other_states"])
        
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
