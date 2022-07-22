import cv2 as cv
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from util import integral_intensity_projection, vertical_line_drawing
from test_lin import get_slope, window_avg, get_valley_window
from scipy.signal import find_peaks

from glob import glob

# def get_flip_point(source):
#     result = np.full(source.shape, False)
#     length = source.shape[0]
#
#     for i in range(length):
#         if i < 1 or i > length - 2:
#             continue
#         base = source[i]
#         result[i] = np.sign(source[i - 1]) != np.sign(base) or np.sign(source[i + 1]) != np.sign(base)
#     return result


image_names = glob('../dentist-CV-yolo/crops/im2*')

if __name__ == '__main__':
    # Plot to see result
    image_paths = glob('../YOLO/crops/*-1-*')
    image_number = 2

    matplotlib.use('module://backend_interagg')
    fig, axs = plt.subplots(image_number, 3)

    for i in range(image_number):
        image_path = image_paths[i]
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        height, width = image.shape
        hor, _ = integral_intensity_projection(image)
        hor = window_avg(hor)
        hor_slope = get_slope(hor.astype('int32'))
        index = np.array(range(height))
        window_position, window_size, valleys = get_valley_window(hor_slope, hor, window_size_0=80, left_margin_0=50)

        print(valleys)

        # row i col 0
        # image = vertical_line_drawing(image, *window_position, color=0)
        axs[i][0].imshow(image, aspect='auto', cmap='gray')
        # for v in valleys:
        #     axs[i][1].axhline(y=v, color='r')

        # row i col 1
        axs[i][1].plot(hor, index, 'g')
        # axs[i][1].xaxis.tick_top()
        axs[i][1].set_ylim(height, 0)
        for v in valleys:
            axs[i][1].axhline(y=v, color='r')

        # row i col 2
        axs[i][2].plot(hor_slope, index, 'g')
        axs[i][2].xaxis.tick_top()
        axs[i][2].set_ylim(height, 0)
        for p in window_position:
            axs[i][2].axhline(y=p, color='r')

    plt.show()
