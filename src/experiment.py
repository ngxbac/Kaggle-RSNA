from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
import random
from catalyst.dl.experiment import ConfigExperiment
from dataset import *
from augmentation import train_aug, valid_aug


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings("ignore")

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module
        #
        # import apex
        # model_ = apex.parallel.convert_syncbn_model(model_)

        return model_

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        """
        image_key: 'id'
        label_key: 'attribute_ids'
        """

        image_size = kwargs.get("image_size", [224, 224])
        train_csv = kwargs.get('train_csv', None)
        valid_csv = kwargs.get('valid_csv', None)
        root = kwargs.get('root', None)

        if train_csv:
            transform = train_aug(image_size)
            train_set = RSNADataset(
                csv_file=train_csv,
                root=root,
                transform=transform
            )
            datasets["train"] = train_set

        if valid_csv:
            transform = valid_aug(image_size)
            valid_set = RSNADataset(
                csv_file=valid_csv,
                root=root,
                transform=transform
            )
            datasets["valid"] = valid_set

        return datasets
