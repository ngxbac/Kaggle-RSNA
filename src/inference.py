import torch
import torch.nn as nn
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
from tqdm import *

from models import *
from augmentation import *
from dataset import *

from sklearn.metrics import log_loss


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = Ftorch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


def predict_test():
    test_csv = "./csv/stage_1_test.csv.gz"
    test_root = "/data/stage_1_test_3w/"

    image_size = [512, 512]
    backbone = "resnet34"
    fold = 0
    scheme = f"{backbone}-mw-512-recheck-{fold}"

    log_dir = f"/logs/rsna/test/{scheme}/"

    with_any = True

    if with_any:
        num_classes = 6
        target_cols = LABEL_COLS
    else:
        num_classes = 5
        target_cols = LABEL_COLS_WITHOUT_ANY

    # test_preds = 0

    model = CNNFinetuneModels(
        model_name=backbone,
        num_classes=num_classes,
        in_chans=3
    )

    ckp = os.path.join(log_dir, f"checkpoints/best.pth")
    checkpoint = torch.load(ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    model = model.to(device)

    print("*" * 50)
    print(f"checkpoint: {ckp}")

    test_dataset = RSNADataset(
        csv_file=test_csv,
        root=test_root,
        with_any=with_any,
        transform=valid_aug(image_size),
        mode="test"
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    test_preds = predict(model, test_loader) / 2

    os.makedirs(f"/logs/prediction/{scheme}", exist_ok=True)
    np.save(f"/logs/prediction/{scheme}/test_{fold}.npy", test_preds)

    test_df = pd.read_csv(test_csv)
    test_ids = test_df['ID'].values

    ids = []
    labels = []
    for i, id in enumerate(test_ids):
        pred = test_preds[i]
        for j, target in enumerate(target_cols):
            id_target = id + "_" + target
            ids.append(id_target)
            labels.append(pred[j])
        if not with_any:
            id_target = id + "_" + "any"
            ids.append(id_target)
            labels.append(pred.max())

    submission_df = pd.DataFrame({
        'ID': ids,
        'Label': labels
    })

    submission_df.to_csv(f"/logs/prediction/{scheme}/{scheme}.csv", index=False)


def predict_test_tta_ckp():
    test_csv = "./csv/stage_1_test.csv.gz"
    test_root = "/data/stage_1_test_3w/"

    image_size = [512, 512]
    backbone = "resnet50"
    fold = 0
    scheme = f"{backbone}-mw-512-recheck-{fold}"

    log_dir = f"/logs/rsna/test/{scheme}/"

    with_any = True

    if with_any:
        num_classes = 6
        target_cols = LABEL_COLS
    else:
        num_classes = 5
        target_cols = LABEL_COLS_WITHOUT_ANY

    # test_preds = 0

    test_preds_all = 0

    for ckp in [7, 8, 9, 10, 11]:
        model = CNNFinetuneModels(
            model_name="resnet50",
            num_classes=num_classes,
            in_chans=3
        )

        ckp = os.path.join(log_dir, f"checkpoints/train512.{ckp}.pth")
        checkpoint = torch.load(ckp)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = nn.DataParallel(model)
        model = model.to(device)

        print("*" * 50)
        print(f"checkpoint: {ckp}")

        augs = test_tta(image_size)

        test_preds = 0

        for name, aug in augs.items():
            print("Augmentation: {}".format(name))

            test_dataset = RSNADataset(
                csv_file=test_csv,
                root=test_root,
                with_any=with_any,
                transform=aug,
                mode="test"
            )

            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=8,
            )

            test_preds += predict(model, test_loader) / 2

        test_preds_all += test_preds / 5

    test_preds = test_preds_all

    os.makedirs(f"/logs/prediction/{scheme}", exist_ok=True)
    np.save(f"/logs/prediction/{scheme}/test_{fold}.npy", test_preds)

    test_df = pd.read_csv(test_csv)
    test_ids = test_df['ID'].values

    ids = []
    labels = []
    for i, id in enumerate(test_ids):
        pred = test_preds[i]
        for j, target in enumerate(target_cols):
            id_target = id + "_" + target
            ids.append(id_target)
            labels.append(pred[j])
        if not with_any:
            id_target = id + "_" + "any"
            ids.append(id_target)
            labels.append(pred.max())

    submission_df = pd.DataFrame({
        'ID': ids,
        'Label': labels
    })

    submission_df.to_csv(f"/logs/prediction/{scheme}/{scheme}_tta_ckp.csv", index=False)


def predict_tta_window():
    test_csv = "./csv/stage_1_test.csv.gz"


    image_size = [224, 224]
    backbone = "resnet50"
    fold = 0
    scheme = f"{backbone}-rndmw-224-{fold}"

    log_dir = f"/logs/rsna/test/{scheme}/"

    with_any = True

    if with_any:
        num_classes = 6
        target_cols = LABEL_COLS
    else:
        num_classes = 5
        target_cols = LABEL_COLS_WITHOUT_ANY

    model = CNNFinetuneModels(
        model_name=backbone,
        num_classes=num_classes,
        in_chans=3
    )

    ckp = os.path.join(log_dir, f"checkpoints/best.pth")
    checkpoint = torch.load(ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    model = model.to(device)

    print("*" * 50)
    print(f"checkpoint: {ckp}")

    test_preds = 0

    for window in ["bone", "subdual", "brain"]:
        test_root = f"/data/stage_1_test/{window}/"
        test_dataset = RSNADataset(
            csv_file=test_csv,
            root=test_root,
            with_any=with_any,
            transform=valid_aug(image_size),
            mode="test"
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
        )



        test_preds += predict(model, test_loader) / 3

    os.makedirs(f"/logs/prediction/{scheme}", exist_ok=True)
    np.save(f"/logs/prediction/{scheme}/test_{fold}.npy", test_preds)

    test_df = pd.read_csv(test_csv)
    test_ids = test_df['ID'].values

    ids = []
    labels = []
    for i, id in enumerate(test_ids):
        pred = test_preds[i]
        for j, target in enumerate(target_cols):
            id_target = id + "_" + target
            ids.append(id_target)
            labels.append(pred[j])
        if not with_any:
            id_target = id + "_" + "any"
            ids.append(id_target)
            labels.append(pred.max())

    submission_df = pd.DataFrame({
        'ID': ids,
        'Label': labels
    })

    submission_df.to_csv(f"/logs/prediction/{scheme}/{scheme}.csv", index=False)


if __name__ == '__main__':
    predict_test()
    # predict_tta_window()
    # predict_pred()
