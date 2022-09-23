import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.edge import integral_intensity_projection, vertical_line_drawing, horizon_line_drawing, gum_jaw_separation, \
    get_rotation_angle, window_avg, get_valley_window, get_slope, vertical_separation
from glob import glob
from scipy import ndimage, misc

from yolov5.utils.plots import save_one_box

# TODO incisor classification test first
if __name__ == '__main__':
    # image_paths = glob('../YOLO/crops/*-1-*')
    image_paths = glob('../YOLO/crops/upper-*')
    image_number = 5
    # image_number = len(image_paths)

    window_size_0_dict = {
        'incisor': 50,
        'molar': 70
    }

    for i in range(0, image_number):
        image_path = image_paths[i]
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        im_0 = image.copy()
        flag, region, _ = image_path.split('\\')[-1].split('-')
        tooth_type = 'incisor' if region == 1 else 'molar'

        theta = get_rotation_angle(image, flag=flag)
        image = ndimage.rotate(image, theta, reshape=True, cval=255)

        gum_sep_line, jaw_sep_line, hor_valleys, hor = gum_jaw_separation(image, flag=flag)


        if flag == 'upper':
            image_roi = image[gum_sep_line:jaw_sep_line, :]
        elif flag == 'lower':
            image_roi = image[jaw_sep_line:gum_sep_line, :]
        else:
            raise ValueError(f'flag only accept upper or lower but get {flag}.')

        window_position, valleys, ver, ver_slope = vertical_separation(image_roi, flag=flag,
                                                                       tooth_type=tooth_type)

        # Plot to see result
        matplotlib.use('module://backend_interagg')
        fig, axs = plt.subplots(3, 2)
        image_name = image_path.split('\\')[-1]
        fig.suptitle(f'{image_name}, theta={theta}')

        # row 0 col 0
        # image = ndimage.rotate(image, theta, reshape=True, cval=255)
        image = vertical_line_drawing(image, *valleys, color=0)
        image = horizon_line_drawing(image, gum_sep_line, jaw_sep_line, color=0)
        axs[0][0].imshow(image, aspect='auto', cmap='gray')

        # row 1 col 1
        axs[1][1].imshow(im_0, aspect='auto', cmap='gray')

        # row 0 col 1
        height, width = image.shape
        # index = np.array(range(height))
        # axs[0][1].plot(hor, index, 'g')
        # axs[0][1].xaxis.tick_top()
        #
        # axs[0][1].set_ylim(height, 0)
        # for v in hor_valleys:
        #     axs[0][1].axhline(y=v, color='r')

        # row 1 col 0
        axs[1][0].xaxis.set_ticks_position('bottom')
        axs[1][0].set_xlim(xmin=0, xmax=width)
        axs[1][0].plot(ver)

        # row 2 col 0
        axs[2][0].xaxis.set_ticks_position('bottom')
        axs[2][0].set_xlim(xmin=0, xmax=width)
        # axs[2][0].set_ylim(ymin=ver_slope.min(), ymax=ver_slope.max())
        axs[2][0].plot(ver_slope)

        for p in window_position:
            axs[1][0].axvline(x=p, color='r')
            axs[2][0].axvline(x=p, color='r')

        plt.show()
