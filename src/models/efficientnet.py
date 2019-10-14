import torch.nn as nn
import pretrainedmodels
from cnn_finetune import make_model
import timm


class EfficientNet(nn.Module):
    def __init__(self,  model_name="tf_efficientnet_b5",
                        num_classes=1108,
                        n_channels=6):
        super(EfficientNet, self).__init__()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        conv1 = self.model.conv_stem
        self.model.conv_stem = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model.conv_stem.weight.data[:, :3, :, :] = conv1.weight.data
        self.model.conv_stem.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
