import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#https://www.kaggle.com/jackvial/plasticc-pytorch
class LogLoss(nn.Module):
    def __init__(self, weights):
        super(LogLoss, self).__init__()
        self.weights = torch.tensor(weights, requires_grad=False, dtype=torch.float32).cuda()

    def forward(self, outputs, targets):
        """

        :param outputs: are the logits
        :param targets: are one-hot vector
        :return: log loss
        """

        outputs = F.logsigmoid(outputs).float()
        outputs_sum = outputs.sum(dim=0, keepdim=True)

        outputs_sum[outputs_sum == 0] = 1
        outputs = outputs / outputs_sum
        WLL = torch.sum(outputs * targets, dim=0)
        WSUM = torch.sum(self.weights, dim=0)
        loss = -(torch.dot(self.weights, WLL) / WSUM)
        return loss
