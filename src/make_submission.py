import pandas as pd
import numpy as np
import cv2
from tqdm import *


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def resize(valid_preds):
    valid_resized_preds = []

    for valid_pred in valid_preds:
        valid_resized_pred = []
        for pred in valid_pred:
            if pred.shape != (525, 350):
                pred = cv2.resize(pred, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_resized_pred.append(pred)
        valid_resized_pred = np.asarray(valid_resized_pred)
        valid_resized_preds.append(valid_resized_pred)
    valid_resized_preds = np.asarray(valid_resized_preds)
    return valid_resized_preds


def optimize_search(pred, gt):
    attempts = []
    for t in tqdm(range(0, 100, 5)):
        t /= 100
        for ms in [5000, 10000]:
            masks = []
            for pred_ in pred:
                predict, num_predict = post_process(pred_, t, ms)
                masks.append(predict)

            d = []
            for i, j in zip(masks, gt):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                else:
                    d.append(dice(i, j))
            attempts.append((t, ms, np.mean(d)))

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]

    #     class_params[class_id] = (best_threshold, best_size)
    return best_threshold, best_size


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    backbone = "resnet34"
    fold = 0

    valid_preds = np.load(f"./prediction/{backbone}/valid_{fold}.npy")
    valid_gts = np.load(f"./prediction/{backbone}/gts_{fold}.npy")
    test_preds = np.load(f"./prediction/{backbone}/test_{fold}.npy")

    valid_resized_preds = resize(valid_preds)
    valid_resized_gts = resize(valid_gts)
    test_resized_preds = resize(test_preds)

    class_params = {}
    for i in range(4):
        best_threshold, best_size = optimize_search(valid_resized_preds[:, i, ::], valid_resized_gts[:, i, ::])
        class_params[i] = [best_threshold, best_size]

    encoded_pixels = []
    image_id = 0

    for preds in test_resized_preds:
        for i, pred in enumerate(preds):
            t = class_params[i][0]
            ms = class_params[i][1]
            pred2, num_predict = post_process(pred, t, ms)
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(pred2)
                encoded_pixels.append(r)

    sub = pd.read_csv("/raid/data/kaggle/cloud/sample_submission.csv")
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(f'./prediction/{backbone}/submission_{fold}.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
