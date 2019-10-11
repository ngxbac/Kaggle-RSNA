import torch
import torch.nn as nn
from cnn_finetune import make_model
from timm import create_model


def cnnfinetune_freeze(self):
    for param in self.parameters():
        param.requires_grad = False

    for param in self._classifier.parameters():
        param.requires_grad = True


def cnnfinetune_unfreeze(self):
    for param in self.parameters():
        param.requires_grad = True


def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


class MultiModals(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=6, dropout_p=None):
        super(MultiModals, self).__init__()
        self.model = make_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_p=dropout_p,
            # classifier_factory=make_classifier
        )

        in_features = self.model._classifier.in_features

        self._classifier = nn.Sequential(
            nn.Linear(in_features + 8, 512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        setattr(self, 'freeze', cnnfinetune_freeze)
        setattr(self, 'unfreeze', cnnfinetune_unfreeze)

    def forward(self, images, meta):
        x = self.model._features(images)
        x = self.model.pool(x)
        x = x.view(x.size(0), -1)
        # import pdb
        # pdb.set_trace()
        # if isinstance(x, torch.HalfTensor):
        #     meta = meta.half()
        x = torch.cat([x, meta], dim=1)
        return self._classifier(x)
