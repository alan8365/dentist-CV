import numpy as np


def integral_intensity_projection(image):
    ver = np.sum(image, axis=0).astype('int32')
    hor = np.sum(image, axis=1).astype('int32')

    return hor, ver
