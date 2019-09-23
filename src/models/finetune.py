from cnn_finetune import make_model
from timm import create_model


def TIMMModels(model_name, pretrained=True, num_classes=6, in_chans=3):
    model = create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
    )

    return model


def CNNFinetuneModels(model_name, pretrained=True, num_classes=6, dropout_p=None):
    model = make_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
    )

    return model
