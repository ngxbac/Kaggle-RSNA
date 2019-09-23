
#!/usr/bin/env python

"""
Stochastic Weight Averaging (SWA)
Averaging Weights Leads to Wider Optima and Better Generalization
https://github.com/timgaripov/swa
"""
import torch
import models
from tqdm import tqdm
import glob


def moving_average(net1, net2, alpha=1.):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    pbar = tqdm(loader, unit="images", unit_scale=loader.batch_size)
    for batch in pbar:
        input, targets = batch['images'], batch['targets']
        input = input.cuda()
        b = input.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from torchvision.transforms import Compose
    from torch.utils.data import DataLoader
    from augmentation import valid_aug
    from dataset import SIIMDataset

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help='input directory')
    parser.add_argument("--output", type=str, default='swa_model.pth', help='output model file')
    parser.add_argument("--batch-size", type=int, default=16, help='batch size')
    args = parser.parse_args()

    # directory = Path(args.input)
    # files = [f for f in directory.iterdir() if f.suffix == ".pth"]
    files = glob.glob(args.input + "/stage1/checkpoints/stage1.*.pth")
    files += glob.glob(args.input + "/stage2/checkpoints/stage1.*.pth")
    assert(len(files) > 1)

    net = models.Unet(
        encoder_name="resnet34",
        activation='sigmoid',
        classes=1,
        # center=True
    )
    checkpoint = torch.load(files[0])
    net.load_state_dict(checkpoint['model_state_dict'])

    for i, f in enumerate(files[1:]):
        # net2 = model.load(f)
        net2 = models.Unet(
            encoder_name="resnet34",
            activation='sigmoid',
            classes=1,
            # center=True
        )
        checkpoint = torch.load(f)
        net2.load_state_dict(checkpoint['model_state_dict'])
        moving_average(net, net2, 1. / (i + 2))

    test_csv = './csv/train_0.csv'
    root = "/raid/data/kaggle/siim/siim256/"
    # img_size = 128
    batch_size = 16
    train_transform = valid_aug()
    train_dataset = SIIMDataset(
        csv_file=test_csv,
        root=root,
        transform=train_transform,
        mode='train'
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    net.cuda()
    bn_update(train_dataloader, net)

    # models.save(net, args.output)
    torch.save({
        'model_state_dict': net.state_dict()
    }, args.output)
