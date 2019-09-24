import numpy as np
import torch
import torch.nn as nn


class LogLoss(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):

        if weight is None:
            pass
        else:
            self.weight = torch.tensor(weight, requires_grad=False, dtype=torch.float32).cuda()

        super(LogLoss, self).__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )
