# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from losses import *
from callbacks import *
from optimizers import *

# Register models
registry.Model(CNNFinetuneModels)
registry.Model(TIMMModels)
registry.Model(MultiModals)

# Register callbacks
registry.Callback(MultiTaskCriterionCallback)

registry.Criterion(LogLoss)

registry.Optimizer(Nadam)
registry.Optimizer(AdamW)