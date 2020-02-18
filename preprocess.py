import numpy as np
import cv2


def scale_to_fit(image, **kwargs):
    image_size = kwargs['image_size']
    image_shape = image.shape
    scale = min(image_size[0] / image_shape[0],
                image_size[1] / image_shape[1])
    m = np.asarray([[scale, 0, 0], [0, scale, 0]],
                   dtype=np.float32)
    image = cv2.warpAffine(image, m, tuple(image_size))

    return image
