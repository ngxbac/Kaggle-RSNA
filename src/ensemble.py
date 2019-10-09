import os
import pandas as pd
import numpy as np
from dataset import LABEL_COLS


if __name__ == '__main__':
    target_cols = LABEL_COLS
    test_csv = "./csv/stage_1_test.csv.gz"
    pred_paths = [
        '/logs/prediction/resnet34-mw-512-recheck-0/test_0.npy',
        '/logs/prediction/resnet50-mw-512-recheck-0/test_0.npy',
        # '/logs/predictions/se_resnext50_32x4d-mw-512-recheck-0/test_0.npy',
    ]

    test_preds = 0
    for pred in pred_paths:
        test_preds += np.load(pred)

    test_preds = test_preds / len(pred_paths)

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
        # if not with_any:
        #     id_target = id + "_" + "any"
        #     ids.append(id_target)
        #     labels.append(pred.max())

    submission_df = pd.DataFrame({
        'ID': ids,
        'Label': labels
    })

    os.makedirs(f"/logs/prediction/ensemble/", exist_ok=True)

    submission_df.to_csv(f"/logs/prediction/ensemble/r34_r50_0.csv", index=False)
