import cv2
import numpy as np
import random
from PIL import Image

import preprocess
from AutoAugment import autoaugment


class NailsDataset:

    def __init__(self, files, transforms):
        self.files = files
        self.transforms = transforms
        self.aug_policy = autoaugment.ImageNetPolicy()

    def __getitem__(self, index):
        file = self.files[index]
        image = cv2.imread(file)
        # TODO: preprocess..

        for fn_name, kwargs in self.transforms.items():
            fn = getattr(preprocess, fn_name)
            image = fn(image, **kwargs)

        label = 1 if '_good' in file else 0

        if random.randint(0, 1) == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.aug_policy(image)
            image = np.asarray(image)
        image = (image / 255.) - 0.5
        image = image.astype(np.float32)
        return {'image': image}, {'label': label}

    def __iter__(self):
        indexes = range(len(self.files))
        return iter(indexes)

    def __len__(self):
        return len(self.files)
