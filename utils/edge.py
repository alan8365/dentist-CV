from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from scipy import ndimage
from scipy import special
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from skimage import color
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from yolov5.utils.plots import Annotator

from utils.general import integral_intensity_projection
from utils.preprocess import recovery_rotated_bounding
from utils.yolo import get_teeth_ROI, crop_by_xyxy

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


def consecutive(data, step_size=1):
    """
    This function takes a list of numbers and splits it into sub-arrays of consecutive numbers.
    Args:
        data: A 1-D array of numerical data
        step_size: The size of the step between consecutive elements in the array (default is 1)

    Returns: An array of sub-arrays, each containing consecutive elements from the original array
    """
    return np.split(data, np.where(np.diff(data) != step_size)[0] + 1)


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


def get_slope(source):
    """
    Args:
        source: The source array to get the slope from

    Returns:
        An array of the same size as the input containing the slope of the source array at each index
    """
    x = np.array(range(source.shape[0]))

    result = np.gradient(source, x)

    return result


# FIXME molar size change process
# FIXME valley cross the tooth
def get_valley_window(source, window_size_0=50, left_margin_0=50):
    hor, ver = integral_intensity_projection(source)
    ver = window_avg(ver)
    ver_slope = get_slope(ver)

    integral = ver
    slope = ver_slope

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

    # last window may not include last valley
    last_window_size = length - window_position[-1]
    if last_window_size > window_size_0 // 2:
        window_position.append(length - 1)
        window_size.append(last_window_size)

    # left margin forward too much
    if window_position[0] > left_margin_0:
        window_position = np.insert(window_position, 0, 0)
        window_size = np.insert(window_size, 0, window_position[1])

    # intensity segment will large then 1 if horizon intensity has huge change.
    intensity_segment = len(consecutive(np.where(hor < hor.max() * 0.8)[0]))
    valleys = []
    for i in range(len(window_size)):
        start = window_position[i]
        end = window_position[i + 1]

        # Peak finding
        has_significant_peak = False

        peaks, properties = find_peaks(integral[start:end], height=0)
        peaks += start
        if len(peaks) > 0:
            peak = peaks[integral[peaks].argmax()]
            if len(peaks) > 1:
                peaks_diff = integral[peaks] - integral[peak]
                has_significant_peak = np.logical_or(peaks_diff < -100, peaks_diff == 0).all()
            else:
                has_significant_peak = True

        if has_significant_peak and intensity_segment == 1:
            # TODO check start is necessary
            zero_point_near_peak = np.abs(integral[start:peak]).argmin() + start
            valleys.append(zero_point_near_peak)
        else:
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


def gum_jaw_separation(source, flag='upper', margin=30, padding=200):
    if flag == 'lower':
        source = np.flip(source, axis=0)

    source = source[padding:, :]
    source = cv.equalizeHist(source)

    hor, _ = integral_intensity_projection(source)
    hor = window_avg(hor)

    hor_valleys, _ = find_peaks(hor * -1)

    height, width = source.shape

    jaw_sep_line = height
    gum_sep_line = 0
    default_return = [gum_sep_line, jaw_sep_line, hor_valleys, hor]

    # 假如沒有任何波谷則不放大
    if hor_valleys.size > 0:
        jaw_sep_line = hor_valleys[hor[hor_valleys].argmin()]

        gum_sep_line_pool = hor_valleys[hor_valleys < jaw_sep_line - 30]
        if gum_sep_line_pool.size == 0:
            # 根據經驗推估一個可能的位置
            gum_sep_line = jaw_sep_line - 130
        else:
            # before change
            gum_sep_line = gum_sep_line_pool[-1]
            # after change
            # gum_sep_line = gum_sep_line_pool.min()

        gum_sep_line -= margin

    if jaw_sep_line > height // 3:
        default_return[0] = gum_sep_line
        default_return[1] = jaw_sep_line

    if flag == 'lower':
        default_return[0] = height - default_return[0]
        default_return[1] = height - default_return[1]
    else:
        default_return[0] = default_return[0] + padding
        default_return[1] = default_return[1] + padding

    return default_return


def get_rotation_angle(source, flag='upper', tooth_position='middle'):
    if flag == 'upper':
        source = np.flip(source)

    padding = 150
    source = source[:-padding]
    source = np.pad(source, (100, 100), mode='constant', constant_values=255)
    minimum_value = np.Inf
    target_angle = 0

    if tooth_position == 'left':
        angle_range = range(0, 21)
    elif tooth_position == 'right':
        angle_range = range(-20, 1)
    else:
        return target_angle

    for i in angle_range:
        source_r = ndimage.rotate(source, i, reshape=False, cval=255)
        hor, _ = integral_intensity_projection(source_r)

        hor_minimum = hor[:-100].min()

        if hor_minimum < minimum_value:
            target_angle = i
            minimum_value = hor_minimum

    return target_angle


# TODO refined curve
def vertical_separation(source, flag='upper', tooth_type='molar', angle=0):
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
    window_position, window_size, valleys = get_valley_window(source, window_size_0=window_size_0)

    return window_position, valleys, ver, ver_slope


def tooth_isolation(source, flag='upper', tooth_position='left', filename=None, save=False, rotation_fix=False):
    """
    Input image and some args to get the tooth in this region.

    Args:
        source:
        flag:
        tooth_position:
        filename:
        save:
        rotation_fix:

    Returns:
        crop_regions: dict of tooth, and it's location.
        angle: the angle of tooth region image rotate.
    """
    # Find angle to rotate image.
    result = {'crop_regions': {}, 'angle': 0}

    theta = get_rotation_angle(source, flag=flag, tooth_position=tooth_position)
    source_rotated = ndimage.rotate(source, theta, reshape=True, cval=255)
    result['angle'] = theta

    # Find ROI
    gum_sep_line, jaw_sep_line, hor_valleys, hor = gum_jaw_separation(source_rotated, flag=flag)

    height, width = source.shape
    phi = np.radians(abs(theta))
    opposite = np.sin(phi) * width
    adjacent = np.cos(phi) * height

    if flag == 'upper':
        source_roi = source_rotated[gum_sep_line:jaw_sep_line, :]
        y1 = gum_sep_line
        y2 = jaw_sep_line
    elif flag == 'lower':
        source_roi = source_rotated[jaw_sep_line:gum_sep_line, :]
        y1 = jaw_sep_line
        y2 = gum_sep_line
    else:
        raise ValueError(f'flag only accept upper or lower but get {flag}.')

    left_padding, right_padding = 0, 1
    if theta > 0:
        left_padding = round((y2 - opposite) * np.tan(phi))
        right_padding = round((adjacent - y1) * np.tan(phi))
    elif theta < 0:
        left_padding = round((adjacent - y1) * np.tan(phi))
        right_padding = round((y2 - opposite) * np.tan(phi))
    source_roi = source_roi[:, left_padding:-right_padding]

    # Split tooth
    tooth_type = 'incisor' if tooth_position == 'middle' else 'molar'
    window_position, valleys, ver, ver_slope = vertical_separation(source_roi, flag=flag, tooth_type=tooth_type,
                                                                   angle=theta)
    # Missing tooth detection
    ver_mean = np.median(ver)
    tooth_missing_region = []
    for i in range(ver.shape[0] - 1):
        is_derivative_smooth = abs(ver_slope[i] - ver_slope[i + 1]) < 45
        curve_under_mean = ver[i] < ver_mean

        if is_derivative_smooth and curve_under_mean:
            tooth_missing_region.append(i)
    tooth_missing_region = consecutive(tooth_missing_region)
    tooth_missing_region = [(i[0], i[-1]) for i in tooth_missing_region if len(i) > 60]

    # Delete redundant valley
    for missing_region in tooth_missing_region:
        peaks, _ = find_peaks(ver_slope[missing_region[0]:missing_region[1]], height=0)
        if len(peaks) < 1 or ver_slope[peaks + missing_region[0]].max() > 100:
            continue

        valley_between_missing_tooth = ((missing_region[0] < valleys) & (valleys < missing_region[1]))
        valley_near_missing_tooth = np.logical_or(np.abs(valleys - missing_region[0]) < 20,
                                                  np.abs(valleys - missing_region[1]) < 20)
        valley_need_delete = np.logical_or(valley_between_missing_tooth, valley_near_missing_tooth)

        valleys = valleys[~valley_need_delete]
        valleys = np.append(valleys, missing_region)

    valleys.sort()
    bounding_number = len(valleys) - 1

    # Check tooth number
    # FIXME multi bounding problem
    tooth_number_dict = all_tooth_number_dict[flag][tooth_position]
    # if tooth_position == 'middle':
    #     if bounding_number < 4:
    #         return result
    # elif tooth_position == 'left' or tooth_position == 'right':
    #     if bounding_number < 3:
    #         return result
    # else:
    #     raise ValueError(f'tooth_region only accept left, middle or right but get {tooth_position}.')

    if flag == 'upper':
        y1, y2 = gum_sep_line // 2, jaw_sep_line
    elif flag == 'lower':
        y1, y2 = jaw_sep_line, gum_sep_line * 2
    else:
        raise ValueError(f'flag only accept upper or lower but get {flag}.')

    unknown_counter = 50
    crop_regions = {}
    tooth_missing_left_bound = {i[0] for i in tooth_missing_region}

    for i in range(bounding_number):
        is_missing = valleys[i] in tooth_missing_left_bound

        x1 = valleys[i] + left_padding
        x2 = valleys[i + 1] + left_padding
        xyxy = torch.Tensor([x1, y1, x2, y2])

        try:
            tooth_number = tooth_number_dict[i]
        except KeyError:
            tooth_number = unknown_counter
            unknown_counter += 1

        # FIXME recovery_rotated_bounding broken
        if rotation_fix:
            crop_source = source
            shape_h, shape_w = source.shape
            xyxy = np.vstack([xyxy, xyxy])
            xyxy = recovery_rotated_bounding(phi, (shape_w, shape_h), xyxy)[0]
        else:
            crop_source = source_rotated

        save_filename = f'{filename} {tooth_number}'
        save_file = Path(f'./crops/{filename}/{save_filename}.jpg')
        crop_image = crop_by_xyxy(crop_source, xyxy.int(), save=save, file=save_file)
        # crop_image = crop_by_xyxy(crop_source, xyxy.int())
        crop_regions[tooth_number] = {'xyxy': xyxy, 'is_missing': is_missing, 'crop_image': crop_image}

    result['crop_regions'] = crop_regions

    return result


def get_all_teeth(results):
    teeth_roi = get_teeth_ROI(results)

    split_teeth = teeth_roi['split_teeth']
    teeth_roi = teeth_roi['images']

    teeth_region = {}
    for filename, data in teeth_roi.items():
        teeth_region[filename] = {}
        for datum in data:
            flag = datum['flag']
            number = datum['number']
            im = datum['image']
            offset = datum['offset']
            tooth_position = datum['tooth_position']

            im_g = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

            isolation_data = tooth_isolation(im_g, flag=flag, filename=filename, tooth_position=tooth_position)

            # region = isolation_data['crop_regions']

            teeth_region[filename][f'{flag}-{tooth_position}'] = isolation_data

        teeth_region[filename][f'split_teeth'] = {'crop_regions': split_teeth[filename], 'angle': 0}

    return teeth_region


def bounding_teeth_on_origin(results, save=False, rotation_fix=False, yolov8=False):
    teeth_roi = get_teeth_ROI(results, yolov8=yolov8)

    split_teeth = teeth_roi['split_teeth']
    teeth_roi = teeth_roi['images']

    tooth_position_dict = {
        0: 'left',
        1: 'middle',
        2: 'right'
    }

    teeth_region = {}
    for file_name, data in teeth_roi.items():
        temp_region = {}
        for datum in data:
            flag = datum['flag']
            im = datum['image']
            offset = datum['offset']
            tooth_position = datum['tooth_position']

            im_g = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

            isolation_data = tooth_isolation(im_g, flag=flag, filename=file_name, tooth_position=tooth_position,
                                             save=save,
                                             rotation_fix=rotation_fix)
            region = isolation_data['crop_regions']
            for k, v in region.items():
                xyxy = region[k]['xyxy']

                xyxy[0] += offset[0]
                xyxy[1] += offset[1]
                xyxy[2] += offset[0]
                xyxy[3] += offset[1]

                region[k]['xyxy'] = xyxy

            temp_region.update(region)
            temp_region.update(split_teeth[file_name])

            teeth_region[file_name] = temp_region

    # Display bounding area
    for file_name, data in teeth_region.items():
        im0 = cv.imread(f'../Datasets/phase-2/{file_name}.jpg')
        annotator = Annotator(im0, line_width=3, example=file_name)
        for tooth_number, body in data.items():
            xyxy = body['xyxy']
            annotator.box_label(xyxy, str(tooth_number), color=(255, 0, 0))

        im1 = annotator.result()

        plt.imshow(im1)
        plt.title(file_name)
        plt.show()

    return teeth_region


def super_pixel(im_g, num_segments=30):
    im_float = img_as_float(im_g)

    labels = slic(im_float, n_segments=num_segments, compactness=0.3, sigma=5)

    fig = plt.figure("Superpixels -- %d segments" % num_segments)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(1 - im_float, labels))
    plt.axis("off")

    out = color.label2rgb(labels, im_float, kind='avg', bg_label=0)[:, :, 0]

    return out


def fill_rotate(im, im_roi_xyxy, theta, expand_size=None):
    h_0, w_0 = im_roi_xyxy[3] - im_roi_xyxy[1], im_roi_xyxy[2] - im_roi_xyxy[0]

    s, c = special.sindg(abs(theta)), special.cosdg(theta)

    x_expand, y_expand = int(s * c * h_0), int(s * c * w_0)
    if expand_size:
        x_expand += expand_size[0]
        y_expand += expand_size[1]
    im_roi_xyxy_expand = im_roi_xyxy + np.array([-x_expand, -y_expand, x_expand, y_expand])
    x1, y1, x2, y2 = im_roi_xyxy_expand
    im_roi_expand = im[y1:y2, x1:x2]
    if theta == 0:
        return im_roi_expand

    r_e = rotate(im_roi_expand, theta)

    h, w = int(h_0 * c + w_0 * s), int(h_0 * s + w_0 * c)
    h_e, w_e = r_e.shape

    w_padding = (w_e - w) // 2
    h_padding = (h_e - h) // 2
    if expand_size:
        w_padding -= expand_size[0]
        h_padding -= expand_size[1]

    r_e = r_e[h_padding:-h_padding, w_padding:-w_padding]

    return r_e


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
