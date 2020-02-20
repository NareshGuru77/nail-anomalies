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
    # from the color space analysis, brightness channel seems to be a good choice..
    ret, thresh = cv2.threshold(img_hsv[:, :, 2], 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append([box, cv2.contourArea(box), cnt])

    # sort all the non axis aligned boxes of all contours
    # according to the box area in descending order...
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    # the top white region seems to always be the largest..
    # the second largest box is mostly on the nail...
    box = boxes[1]
    rect = cv2.boundingRect(box[2])
    x, y, w, h = rect
    # get the center of the bounding rectangle...
    center = (x + (w / 2), y + (h / 2))
    # pick a larger region which contains this bounding rectangle..
    x, y = int(center[0] - width / 2), int(center[1] - height / 2)
    # prevent negative x and y...
    x = max(0, x)
    y = max(0, y)

    # this image mostly contains the nail...
    # lower dimensional with essential information..
    return img[y:y + height, x:x + width]


def extract_features(img):
    # hog features..
    _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualize=True, multichannel=True)

    return hog_image


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

    # fit an svm...
    clf = svm.SVC()
    clf.fit(train_X, train_y)

    # predict labels of test images..
    preds = clf.predict(test_X)
    accuracy = accuracy_score(test_y, preds)
    print(f'Accuracy of baseline approach {accuracy}')

    cm = confusion_matrix(test_y, preds)
    print(cm)


def train_cnn(config):
    my_learner = Learner(config['model_dir'], **config['kwargs'])
    train_and_eval_kwargs = my_learner.get_train_and_eval_kwargs()
    tf.estimator.train_and_evaluate(**train_and_eval_kwargs)


def test_cnn(config):
    my_learner = Learner(config['model_dir'], **config['kwargs'])
    predict_kwargs = my_learner.get_predict_kwargs(config['checkpoint_path'])
    predictions = my_learner.predict(**predict_kwargs)

    test_files = my_learner.test_dataset().files
    preds = []
    labels = []
    for p, file in zip(predictions, test_files):
        preds.append(p['class'])
        labels.append(1 if 'good' in file else 0)

    accuracy = accuracy_score(labels, preds)
    print(f'Accuracy of cnn approach {accuracy}')

    cm = confusion_matrix(labels, preds)
    print(cm)


def main():
    logging.getLogger().setLevel(logging.INFO)
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./configs/baseline.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    method = config['method']
    getattr(sys.modules[__name__], method)(config)


def get_preds():
    from tensorflow.contrib import predictor
    from preprocess import scale_to_fit
    image = cv2.imread('./data/nailgun/good/1522072665_good.jpeg')
    image = scale_to_fit(image, **{'image_size': [256, 256]})
    image = np.expand_dims(image, axis=0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    predict_fn = predictor.from_saved_model('./models/saved_model/',
                                            config=config)
    predictions = predict_fn({'image': image})
    print(predictions)


if __name__ == '__main__':
    main()
