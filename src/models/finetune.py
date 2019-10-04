import torch.nn as nn
from cnn_finetune import make_model
from timm import create_model


def timm_freeze(self):
    for param in self.parameters():
        param.requires_grad = False

    for param in self.get_classifier().parameters():
        param.requires_grad = True


def timm_unfreeze(self):
    for param in self.parameters():
        param.requires_grad = True


def cnnfinetune_freeze(self):
    for param in self.parameters():
        param.requires_grad = False

    for param in self._classifier.parameters():
        param.requires_grad = True


def cnnfinetune_unfreeze(self):
    for param in self.parameters():
        param.requires_grad = True


def TIMMModels(model_name, pretrained=True, num_classes=6, in_chans=3):
    model = create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )

    conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels=in_chans,
                                 out_channels=conv1.out_channels,
                                 kernel_size=conv1.kernel_size,
                                 stride=conv1.stride,
                                 padding=conv1.padding,
                                 bias=conv1.bias)

    # copy pretrained weights
    model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
    model.conv1.weight.data[:, 3:in_chans, :, :] = conv1.weight.data[:, :int(in_chans - 3), :, :]

    setattr(model, 'freeze', timm_freeze)
    setattr(model, 'unfreeze', timm_unfreeze)

    return model


def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


def CNNFinetuneModels(model_name, pretrained=True, num_classes=6, dropout_p=None, in_chans=3):
    model = make_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
        classifier_factory=make_classifier
    )

    conv1 = model._features[0]
    model._features[0] = nn.Conv2d(in_channels=in_chans,
                                         out_channels=conv1.out_channels,
                                         kernel_size=conv1.kernel_size,
                                         stride=conv1.stride,
                                         padding=conv1.padding,
                                         bias=conv1.bias)

    # copy pretrained weights
    model._features[0].weight.data[:, :3, :, :] = conv1.weight.data
    model._features[0].weight.data[:, 3:in_chans, :, :] = conv1.weight.data[:, :int(in_chans - 3), :, :]

    setattr(model, 'freeze', cnnfinetune_freeze)
    setattr(model, 'unfreeze', cnnfinetune_unfreeze)

    return model
