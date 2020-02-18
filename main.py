import cv2
import numpy as np
import os
from scipy.spatial.distance import cdist
import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def find_areas(cls_files):
    areas = []
    for file in cls_files:
        im = cv2.imread(file)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append([box, cv2.contourArea(box)])

        boxes = sorted(boxes, key=lambda x: x[1])
        box = boxes[-2]
        areas.append(box[1])
        lengths = cdist(box[0], box[0])
        unique_lengths = np.unique(lengths)
        unique_lengths = unique_lengths[unique_lengths > 0]
        aspect_ratio = np.max(unique_lengths) / np.min(unique_lengths)
        cv2.drawContours(im, [box[0]], 0, (0, 0, 255), 2)

    return areas


def process():
    path = './data/nailgun/'
    good_files = glob.glob(os.path.join(path, '**/*_good.jpeg'))
    bad_files = glob.glob(os.path.join(path, '**/*_bad.jpeg'))

    areas = find_areas(good_files) + find_areas(bad_files)
    preds = [1 if ar <= max(areas[:99]) else 0 for ar in areas]
    labels = [1 for _ in range(99)] + [0 for _ in range(99)]

    accuracy = accuracy_score(labels, preds)
    print(f'Accuracy of simple approach {accuracy}')

    cm = confusion_matrix(labels, preds)
    print(cm)


if __name__ == '__main__':
    # TODO: color space analysis
    process()
