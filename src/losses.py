from functools import partial
from catalyst.utils import get_activation_fn
from catalyst.contrib.criterion import LovaszLossBinary, FocalLossBinary

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
