import torch
import torch.nn as nn
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
from tqdm import *

from models import *
from augmentation import *
from dataset import *


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
    test_root = "/data/stage_1_test_images_jpg/"

    image_size = [224, 224]
    backbone = "resnet50"
    fold = 0
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
        transform=valid_aug(image_size)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    test_preds = predict(model, test_loader)

    os.makedirs(f"/logs/prediction/{scheme}", exist_ok=True)
    np.save(f"/logs/prediction/{scheme}/test_{fold}.npy", test_preds)

    test_df = pd.read_csv(test_csv)
    test_ids = test_df['ID'].values
    target_cols = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

    ids = []
    labels = []
    for i, id in enumerate(test_ids):
        pred = test_preds[i]
        for j, target in enumerate(target_cols):
            id_target = id + "_" + target
            ids.append(id_target)
            labels.append(pred[j])

    submission_df = pd.DataFrame({
        'ID': ids,
        'Label': labels
    })

    submission_df.to_csv(f"/logs/prediction/{scheme}/submission_{fold}.csv", index=False)


if __name__ == '__main__':
    predict_test()
