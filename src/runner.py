from typing import Mapping, Any
from catalyst.dl.runner import WandbRunner
from catalyst.dl.core import RunnerState
from catalyst.contrib.optimizers import Lookahead


class ModelRunner(WandbRunner):
    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(batch["images"])
        if isinstance(output, tuple):
            return {
                "logits": output[0],
                "cls_logits": output[1]
            }
        else:
            return {
                "logits": output
            }