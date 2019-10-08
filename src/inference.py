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

    # test_preds = 0

    model = CNNFinetuneModels(
        model_name="resnet50",
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

    test_preds = predict(model, test_loader)

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

    submission_df.to_csv(f"/logs/prediction/{scheme}/{scheme}_512.csv", index=False)


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



def multi_weighted_logloss(y_ohe, y_p, class_weight):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)

    # Transform to log
    y_p_log = np.log(y_p)

    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)

    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)

    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


def predict_pred():
    fold = 0
    test_csv = f"./csv/random_kfold/valid_{fold}.csv.gz"
    test_root = "/data/stage_1_train_images_jpg/"

    image_size = [224, 224]
    backbone = "resnet50"
    scheme = f"{backbone}-baseline-{fold}"

    log_dir = f"/logs/rsna/test/{scheme}/"

    model = TIMMModels(
        model_name=backbone,
        num_classes=6
    )

    ckp = os.path.join(log_dir, "checkpoints/best.pth")
    checkpoint = torch.load(ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    model = model.to(device)

    print("*" * 50)
    print(f"checkpoint: {ckp}")

    test_dataset = RSNADataset(
        csv_file=test_csv,
        root=test_root,
        transform=valid_aug(image_size),
        with_any=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    test_preds = predict(model, test_loader)

    os.makedirs(f"/logs/prediction/{scheme}", exist_ok=True)
    np.save(f"/logs/prediction/{scheme}/valid_{fold}.npy", test_preds)

    valid_df = pd.read_csv(test_csv)
    valid_df = valid_df[~valid_df['ID'].isin(IGNORE_IDS)]
    target_cols = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]
    y_true = valid_df[target_cols].values
    # logloss = log_loss(y_true, test_preds)
    weights = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1
    }
    logloss = multi_weighted_logloss(y_true, test_preds, weights)
    print(f"Log Loss: {logloss}")


if __name__ == '__main__':
    predict_test()
    # predict_tta_window()
    # predict_pred()
