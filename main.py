import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import argparse
import logging
import tensorflow as tf
import yaml
import sys
from skimage.feature import hog
from sklearn import svm

from learner import Learner


def crop_nail(img, width, height):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, thresh = cv2.threshold(img_hsv[:, :, 2], 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append([box, cv2.contourArea(box), cnt])

    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    box = boxes[1]
    rect = cv2.boundingRect(box[2])
    x, y, w, h = rect
    center = (x + (w / 2), y + (h / 2))
    x, y = int(center[0] - width / 2), int(center[1] - height / 2)
    x = max(0, x)
    y = max(0, y)

    return img[y:y + height, x:x + width]


def extract_features(img):
    _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualize=True, multichannel=True)

    return hog_image


def train_cnn(config):
    my_learner = Learner(config['model_dir'], **config['kwargs'])
    train_and_eval_kwargs = my_learner.get_train_and_eval_kwargs()
    tf.estimator.train_and_evaluate(**train_and_eval_kwargs)


def get_xy(files):
    X = np.zeros((len(files), 256 * 256))
    y = []
    for idx, file in enumerate(files):
        image = cv2.imread(file)
        image = crop_nail(image, 256, 256)
        features = extract_features(image)
        features = features.flatten()
        X[idx, :] = features
        label = 1 if 'good' in file else 0
        y.append(label)

    return X, y


def baseline_method(config):
    with open(config['splits'], 'r') as f:
        splits = yaml.load(f)

    train_files = splits['train']
    test_files = splits['test']

    train_X, train_y = get_xy(train_files)
    test_X, test_y = get_xy(test_files)

    clf = svm.SVC()
    clf.fit(train_X, train_y)

    preds = clf.predict(test_X)
    accuracy = accuracy_score(test_y, preds)
    print(f'Accuracy of baseline approach {accuracy}')

    cm = confusion_matrix(test_y, preds)
    print(cm)


def main():
    logging.getLogger().setLevel(logging.INFO)
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./configs/cnn.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    method = config['method']
    getattr(sys.modules[__name__], method)(config)


if __name__ == '__main__':
    main()
