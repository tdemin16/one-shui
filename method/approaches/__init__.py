import torch
from method.approaches.lipschitz import LipschitzRegularization
from method.approaches.meta_unlearn import MetaUnlearn
from method.approaches.pgu import ProjectedGradientUnlearning
from method.approaches.scrub import SCRUB
from method.approaches.bad_teacher import BadTeacher
from method.approaches.ssd import SelectiveSynapseDampening
    

__all__ = [
    "MetaUnlearn",
    "SelectiveSynapseDampening",
    "LipschitzRegularization",
    "ProjectedGradientUnlearning",
    "SCRUB",
    "BadTeacher",
]