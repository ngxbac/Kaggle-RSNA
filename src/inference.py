import torch
import torch.nn as nn
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
from tqdm import *

from models import *
from augmentation import *
from dataset import *
import glob

from sklearn.metrics import log_loss


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            # meta = dct["meta"].to(device)
            pred = model(images)
            pred = Ftorch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


def predict_test():
    test_csv = "./csv/patient2_kfold/test.csv"
    # test_root = "/data/stage_1_test_3w/"
    test_root = "/data/png/test_stage_1/adjacent-brain-cropped/"

    image_size = [512, 512]
    backbone = "resnet50"
    # fold = 2
    for fold in [1]:
        #/logs/rsna/test/resnet50-anju-512-resume-0/checkpoints//train512.13.pth
        scheme = f"{backbone}-anjuu-512-{fold}"

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
        )

        ckp = os.path.join(log_dir, f"checkpoints/best.pth")
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

        os.makedirs(f"/logs/prediction/{scheme}", exist_ok=True)
        np.save(f"/logs/prediction/{scheme}/test_{fold}_tta.npy", test_preds)

        test_df = pd.read_csv(test_csv)
        test_ids = test_df['sop_instance_uid'].values

        ids = []
        labels = []
        for i, id in enumerate(test_ids):
            if not "ID" in id:
                id = "ID_" + id
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

        submission_df.to_csv(f"/logs/prediction/{scheme}/{scheme}_tta.csv", index=False)


def get_best_checkpoints(checkpoint_dir, n_best=3, minimize_metric=True):
    files = glob.glob(f"{checkpoint_dir}/checkpoints/train512.*.pth")
    files = [file for file in files if not 'full' in file]
    import pdb
    pdb.set_trace()

    top_best_metrics = []
    for file in files:
        ckp = torch.load(file)
        valid_metric = ckp['valid_metrics']['loss']
        top_best_metrics.append((file, valid_metric))

    top_best_metrics = sorted(
        top_best_metrics,
        key=lambda x: x[1],
        reverse=not minimize_metric
    )
    top_best_metrics = top_best_metrics[:n_best]
    return top_best_metrics


def predict_test_tta_ckp():
    test_csv = "./csv/patient2_kfold/test.csv"
    # test_root = "/data/stage_1_test_3w/"
    test_root = "/data/png/test_stage_1/adjacent-brain-cropped/"

    image_size = [512, 512]
    backbone = "resnet50"
    # fold = 2
    for fold in [1]:
        # /logs/rsna/test/resnet50-anju-512-resume-0/checkpoints//train512.13.pth
        scheme = f"{backbone}-anjuu-512-{fold}"

        log_dir = f"/logs/rsna/test/{scheme}/"

        with_any = True

        if with_any:
            num_classes = 6
            target_cols = LABEL_COLS
        else:
            num_classes = 5
            target_cols = LABEL_COLS_WITHOUT_ANY

        # test_preds = 0

        top_best_metrics = get_best_checkpoints(log_dir, n_best=3, minimize_metric=True)

        test_preds = 0
        for best_metric in top_best_metrics:

            checkpoint_path, checkpoint_metric = best_metric
            print("*" * 50)
            print(f"checkpoint: {checkpoint_path}")
            print(f"Metric: {checkpoint_metric}")

            model = CNNFinetuneModels(
                model_name=backbone,
                num_classes=num_classes,
            )

            ckp = os.path.join(log_dir, f"checkpoints/best.pth")
            checkpoint = torch.load(ckp)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = nn.DataParallel(model)
            model = model.to(device)

            augs = test_tta(image_size)

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
                    batch_size=64,
                    shuffle=False,
                    num_workers=8,
                )

                test_preds += predict(model, test_loader) / (len(augs) * len(top_best_metrics))

        os.makedirs(f"/logs/prediction/{scheme}", exist_ok=True)
        np.save(f"/logs/prediction/{scheme}/test_{fold}_ckp_tta.npy", test_preds)

        test_df = pd.read_csv(test_csv)
        test_ids = test_df['sop_instance_uid'].values

        ids = []
        labels = []
        for i, id in enumerate(test_ids):
            if not "ID" in id:
                id = "ID_" + id
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

        submission_df.to_csv(f"/logs/prediction/{scheme}/{scheme}_ckp_tta.csv", index=False)


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
    # predict_test()
    predict_test_tta_ckp()
