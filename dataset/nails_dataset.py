import cv2
import sys

import preprocess


class NailsDataset:

    def __init__(self, files, transforms):
        self.files = files
        self.transforms = transforms

    def __getitem__(self, index):
        file = self.files[index]
        data = cv2.imread(file)
        # TODO: preprocess..

        for fn_name, kwargs in self.transforms:
            fn = getattr(sys.modules[preprocess], fn_name)
            data = fn(data, **kwargs)

        label = 1 if '_good' in file else 0

        return data, label

    def __iter__(self):
        indexes = range(len(self.files))
        return iter(indexes)

    def __len__(self):
        return len(self.files)
