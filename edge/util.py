import cv2 as cv
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import math

from glob import glob
from PIL import Image, ImageOps
from scipy import ndimage
from pathlib import Path

from yolov5.utils.plots import save_one_box

all_tooth_number_dict = {
    'upper': {
        'left': {0: 16, 1: 15, 2: 14},
        'middle': {0: 12, 1: 11, 2: 21, 3: 22},
        'right': {0: 24, 1: 25, 2: 26},
    },
    'lower': {
        'left': {0: 46, 1: 45, 2: 44},
        'middle': {0: 42, 1: 41, 2: 31, 3: 32},
        'right': {0: 34, 1: 35, 2: 36},
    }
}


def vertical_line_drawing(src, *args, color=0, thickness=3):
    height, width = src.shape
    result = src.copy()

    for i in args:
        result = cv.line(result, (i, 0), (i, height), color, thickness)

    return result


def horizon_line_drawing(src, *args, color=0, thickness=3):
    height, width = src.shape
    result = src.copy()

    for i in args:
        result = cv.line(result, (0, i), (width, i), color, thickness)

    return result


def edge_filter(src, vertical=1, horizon=1):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    # Check if image is loaded fine
    if src is None:
        print('Error opening image')
        return -1

    src = cv.GaussianBlur(src, (3, 3), 0)

    # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = src

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, vertical, abs_grad_y, horizon, 0)
    # blur = cv.bilateralFilter(grad, 33, 33, 35)

    # plt.imshow(grad)
    # plt.show()

    # cv.imshow(window_name, grad)
    # cv.waitKey(0)

    return grad
    # return blur


def blur_pooling(filename):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         ])

    image = Image.open(filename)
    image = ImageOps.grayscale(image)
    image = transform(image)
    # image = image * 255
    grad = edge_filter(image.permute(1, 2, 0).numpy() * 255, vertical=1, horizon=0)
    blur = cv.bilateralFilter(grad, 33, 33, 35)
    # blur = grad

    x = transform(blur)
    pool = nn.MaxPool2d(3, 3)

    output = pool(x)

    # plt.imshow(grad, cmap='gray')
    # plt.title('grad')
    # plt.show()
    #
    # plt.imshow(blur, cmap='gray')
    # plt.title('blur')
    # plt.show()
    #
    # plt.imshow(output.permute(1, 2, 0), cmap='gray')
    # plt.title('output')
    # plt.show()

    return output


def integral_intensity_projection(image):
    ver = np.sum(image, axis=0).astype('int32')
    hor = np.sum(image, axis=1).astype('int32')

    return hor, ver


def window_avg(source, window_size=5):
    result = source.copy().astype('int32')

    pre_index = window_size // 2
    post_index = pre_index + 1

    for i in range(len(source)):
        if i < pre_index:
            pass
            # result[i] = np.sum(source[:i + 1]) / i
        elif i > len(source) - post_index:
            pass
            # result[i] = np.sum(source[i:]) / result.shape[0] - i
        else:
            result[i] = np.average(source[i - pre_index:i + post_index])

    return result


def get_slope(source, window_size=5):
    result = np.zeros(source.shape, dtype='int32')
    step = window_size // 2
    length = len(source)

    for i in range(length):
        if i < step:
            pass
        elif i > length - step - 1:
            pass
        else:
            slope = (source[i + step] - source[i - step]) / window_size
            result[i] = slope

    return result


# FIXME molar size change process
# FIXME valley cross the tooth
def get_valley_window(slope, integral, source=None, window_size_0=50, left_margin_0=50):
    length = slope.shape[0]
    negative_slope_index = np.where(slope < 0)[0]
    positive_slope_index = np.where(slope > 0)[0]

    # find first sign change
    left_margin = length
    for j in negative_slope_index:
        if slope[j - 1] > 0:
            left_margin = j
            break

    # find window size
    window_size = []
    window_position = [left_margin]
    # S = []

    i = 0  # window number
    j = left_margin
    while j < length:
        j += window_size_0

        if j > length - 5:
            break

        # TODO overflow exception
        summation_i = np.sum(slope[j:j + 5]) / 5
        # S.append(S_i)

        # rule a
        if summation_i > 0:
            target_list = negative_slope_index[negative_slope_index > j]
            if len(target_list) == 0:
                break
            j = target_list[0]
        # rule b
        else:
            target_list = positive_slope_index[positive_slope_index < j]
            if len(target_list) == 0:
                break
            j = target_list[-1]

        window_size_current = j - window_position[i]
        if window_size_current < window_size_0:
            window_size_current = window_size_0

        j = window_size_current + window_position[i]

        window_position.append(j)
        window_size.append(window_size_current)

        i += 1

    last_window_size = length - window_position[-1]
    if last_window_size > window_size_0:
        window_position.append(length - 1)
        window_size.append(last_window_size)

    # left margin forward too much
    if window_position[0] > left_margin_0:
        window_position = np.insert(window_position, 0, 0)
        window_size = np.insert(window_size, 0, window_position[1])

    valleys = []
    for i in range(len(window_size)):
        start = window_position[i]
        end = window_position[i + 1]

        valley_position = integral[start:end + 1].argmin() + start
        valleys.append(valley_position)

    # non valley check
    # calculate distance between two valley
    valleys = np.array(valleys)
    valleys_gap = np.array([valleys[i] - valleys[i - 1] for i in range(1, len(valleys))])
    near_valleys = np.where(valleys_gap < window_size_0 // 2)[0]

    valleys_mask = np.full(len(valleys), True)
    for v_i in near_valleys:
        v_1 = valleys[v_i]
        v_2 = valleys[v_i + 1]

        i_1 = integral[v_1]
        i_2 = integral[v_2]

        if i_1 < i_2:
            valleys_mask[v_i + 1] = False
        else:
            valleys_mask[v_i] = False

    valleys = valleys[valleys_mask]

    window_position = np.array(window_position)
    window_size = np.array(window_size)
    return window_position, window_size, valleys


def gum_jaw_separation(source, flag='upper'):
    margin = 30
    theta, hor = get_rotation_angle(source, flag=flag)
    source = ndimage.rotate(source, theta, reshape=True, cval=255)

    hor, _ = integral_intensity_projection(source)
    hor = window_avg(hor)
    hor_slope = get_slope(hor)
    _, _, hor_valleys = get_valley_window(hor_slope, hor, window_size_0=50, left_margin_0=30)

    jaw_sep_line = hor_valleys[hor[hor_valleys].argmin()]
    if flag == 'upper':
        # FIXME IndexError: index -1 is out of bounds for axis 0 with size 0
        gum_sep_line = hor_valleys[hor_valleys < jaw_sep_line - 30][-1]
        gum_sep_line -= margin
    elif flag == 'lower':
        gum_sep_line = hor_valleys[hor_valleys > jaw_sep_line + 30][0]
        gum_sep_line += margin
    else:
        raise ValueError(f'flag only accept upper or lower but get {flag}.')

    return gum_sep_line, jaw_sep_line, theta, hor_valleys, hor


def get_rotation_angle(source, flag='upper'):
    source = np.pad(source, (100, 100), mode='constant', constant_values=255)
    minimum_value = np.Inf
    target_angle = 0
    result_hor = []

    y = 0
    for i in range(-20, 21):
        source_r = ndimage.rotate(source, i, reshape=False, cval=255)
        hor, _ = integral_intensity_projection(source_r)

        if flag == 'upper':
            hor_minimum = hor[100:].min()
        elif flag == 'lower':
            hor_minimum = hor[:-100].min()
        else:
            raise ValueError(f'flag only accept upper or lower but get {flag}.')

        if hor_minimum < minimum_value:
            target_angle = i
            minimum_value = hor_minimum
            result_hor = hor

        # Plot every step of rotation
        # fig, axs = plt.subplots(1, 2)
        # fig.suptitle(f'theta = {i}, value={minimum_value}')
        #
        # axs[0].imshow(source_r, aspect='auto', cmap='gray')
        #
        # height, width = source_r.shape
        # index = np.array(range(height))
        # axs[1].plot(hor, index, 'g')
        # axs[1].xaxis.tick_top()
        #
        # axs[1].set_ylim(height, 0)
        #
        # plt.show()

    return target_angle, result_hor


# TODO different window_size for molar and pre-molar
# TODO refined curve
def vertical_separation(source, flag='upper', tooth_type='molar'):
    window_size_0_dict = {
        'incisor': 40,
        # TODO test window_size_0 change by angle
        'molar': 70
    }

    window_size_0 = window_size_0_dict[tooth_type]

    # vertical segmentation
    # TODO maybe try to combine bounding ?
    _, ver = integral_intensity_projection(source)
    ver = window_avg(ver)
    ver_slope = get_slope(ver)
    window_position, window_size, valleys = get_valley_window(ver_slope, ver, window_size_0=window_size_0)

    return window_position, valleys, ver, ver_slope


def tooth_isolation(source, flag='upper', tooth_position='left', file_name=None, save=False):
    # Find jaw separation line and gums separation
    # FIXME jaw_spe_line in lower missing
    crop_images = {}
    gum_sep_line, jaw_sep_line, theta, hor_valleys, hor = gum_jaw_separation(source, flag=flag)
    if tooth_position == 'middle':
        theta = 0
    else:
        source = ndimage.rotate(source, theta, reshape=True, cval=255)

    if flag == 'upper':
        source_roi = source[gum_sep_line:jaw_sep_line, :]
    elif flag == 'lower':
        source_roi = source[jaw_sep_line:gum_sep_line, :]
    else:
        raise ValueError(f'flag only accept upper or lower but get {flag}.')

    tooth_type = 'incisor' if tooth_position == 'middle' else 'molar'
    window_position, valleys, ver, ver_slope = vertical_separation(source_roi, flag=flag, tooth_type=tooth_type)
    bounding_number = len(valleys) - 1

    # Check tooth number
    # TODO change it to contain missing tooth and fix multi bounding
    tooth_number_dict = all_tooth_number_dict[flag][tooth_position]
    if tooth_position == 'middle':
        if bounding_number < 4:
            return crop_images
    elif tooth_position == 'left' or tooth_position == 'right':
        if bounding_number < 3:
            return crop_images
    else:
        raise ValueError(f'tooth_region only accept left, middle or right but get {tooth_position}.')

    if flag == 'upper':
        y1, y2 = gum_sep_line // 2, jaw_sep_line
    elif flag == 'lower':
        # FIXME gum_sep_line too deep
        y1, y2 = jaw_sep_line, gum_sep_line * 2
    else:
        raise ValueError(f'flag only accept upper or lower but get {flag}.')

    unknown_counter = 50
    source_rgb = cv.cvtColor(source, cv.COLOR_GRAY2RGB)
    for i in range(bounding_number):
        x1 = valleys[i]
        x2 = valleys[i + 1]
        xyxy = [x1, y1, x2, y2]

        try:
            tooth_number = tooth_number_dict[i]
        except KeyError:
            tooth_number = unknown_counter
            unknown_counter += 1

        if save:
            save_file = Path(f'./crops/{file_name}-{tooth_number}.jpg')
            im = save_one_box(xyxy, source_rgb, save=True, file=save_file)
            print(f'Tooth crop saved: {file_name}:{tooth_number}')

        # TODO Rotation offset fix
        theta = math.radians(theta)
        y1_offset = np.tan(theta) * x1
        y2_offset = np.tan(theta) * x2

        region = {tooth_number: {'xyxy': xyxy, 'rotation_offset': [y1_offset, y2_offset]}}

        crop_images.update(region)

    return crop_images


if __name__ == '__main__':
    # image = cv.imread(r'PANO_1~630/20200825090819526_0811001A.jpg', cv.IMREAD_COLOR)
    # image_paths = glob('../YOLO/crops/upper-*')
    # image_number = 5
    #
    # for i in range(0, image_number):
    #     image_path = image_paths[i]
    #     file_name = image_path.split('\\')[-1]
    #     image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    #     im_0 = image.copy()
    #     flag, area, _ = file_name.split('-')
    #     tooth_type = 'incisor' if area == 1 else 'molar'
    #
    #     tooth_isolation(image, flag=flag, tooth_position=tooth_type, file_name=file_name)
    pass
