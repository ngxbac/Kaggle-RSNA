from typing import Mapping, Any
import torch.nn as nn
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.core import RunnerState
from catalyst.contrib.optimizers import Lookahead


class ModelRunner(SupervisedRunner):
    def __init__(
            self,
            model: nn.Module = None,
            device=None,
            input_key: str = "images",
            output_key: str = "logits",
            input_target_key: str = "targets",
    ):
        super(ModelRunner, self).__init__(
            model=model,
            device=device,
            input_key=input_key,
            output_key=output_key,
            input_target_key=input_target_key
        )
