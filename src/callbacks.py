from catalyst.dl.core import Callback, RunnerState, CallbackOrder
from catalyst.dl.callbacks import CriterionCallback
from catalyst.contrib.criterion import FocalLossBinary
from catalyst.dl.utils.criterion import accuracy
from catalyst.utils import get_activation_fn
import torch
import torch.nn as nn
import numpy as np
from typing import List

import torch
from catalyst.utils import get_activation_fn


class MultiTaskCriterionCallback(Callback):
    def __init__(
        self,
        input_seg_key: str = "targets",
        input_cls_key: str = "labels",
        output_seg_key: str = "logits",
        output_cls_key: str = "cls_logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
        loss_weights: List[float] = None,
    ):
        super(MultiTaskCriterionCallback, self).__init__(CallbackOrder.Criterion)
        self.input_seg_key = input_seg_key
        self.input_cls_key = input_cls_key
        self.output_seg_key = output_seg_key
        self.output_cls_key = output_cls_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier
        self.loss_weights = loss_weights

        self.criterion_cls = nn.BCEWithLogitsLoss()

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        output_seg = state.output[self.output_seg_key]
        output_cls = state.output[self.output_cls_key]
        input_seg = state.input[self.input_seg_key]
        input_cls = state.input[self.input_cls_key]

        # assert len(self.loss_weights) == len(outputs)
        loss = 0

        # Segmentation loss
        loss += criterion(output_seg, input_seg) * self.loss_weights[0]
        # Classification loss
        loss += self.criterion_cls(output_cls, input_cls) * self.loss_weights[1]

        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        criterion = state.get_key(
            key="criterion", inner_key=self.criterion_key
        )

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(
            metrics_dict={
                self.prefix: loss.item(),
            }
        )

        self._add_loss_to_state(state, loss)
