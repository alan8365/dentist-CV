import json

import cv2 as cv
import numpy as np
import torch
from scipy import ndimage
from yolov5.models.common import Detections

from utils.edge import get_rotation_angle, gum_jaw_separation
from utils.yolo import get_teeth_ROI


def get_fake_result(image_name):
    img = cv.imread(image_name)
    label_name = image_name.with_suffix('.json')

    with open(label_name, 'r') as f:
        label_file = json.load(f)

    target_list = [''.join(i) for i in zip('11223344', '37' * 4)]
    names = {i: v for i, v in enumerate(target_list)}
    names_reverse = {v: i for i, v in enumerate(target_list)}

    xyxy = []
    for shape in label_file['shapes']:
        label = shape['label']
        if label not in target_list:
            continue

        temp = np.hstack(shape['points'] + [1] + [names_reverse[label]])
        temp = torch.from_numpy(temp)
        xyxy.append(temp)

    imgs = [img]
    pred = [torch.stack(xyxy)]
    files = [image_name.name]

    result = Detections(
        imgs=imgs,
        pred=pred,
        files=files,
        names=names,
    )

    return result


def quick_get_roi(image_name=None, model=None, roi_index=0, random_sample=False, image_dir=None):
    if random_sample:
        assert (image_dir is not None), "Random sample but don't give image_dir"

        image_names = list(image_dir.glob('*.jpg'))
        random_image_index = np.random.randint(0, len(image_names))
        image_name = image_names[random_image_index]
        roi_index = np.random.randint(0, 7)

    assert (image_name is not None), "Not random sample and don't give image_name"

    filename = image_name.stem

    if model:
        results = model(image_name)
    else:
        results = get_fake_result(image_name)

    teeth_roi = get_teeth_ROI(results)
    teeth_roi_images = teeth_roi['images'][filename]
    teeth_roi_split_teeth = teeth_roi['split_teeth']

    target_roi = teeth_roi_images[roi_index]
    target_roi_image = target_roi['image']
    flag = target_roi['flag']
    tooth_position = target_roi['tooth_position']
    im_g = cv.cvtColor(target_roi_image, cv.COLOR_RGBA2GRAY)
    im_g_shape = np.array(np.array(im_g.shape)[[1, 0]])

    return im_g, flag, tooth_position, teeth_roi_split_teeth, target_roi


def quick_rotate_and_zooming(source, flag, tooth_position):
    theta = get_rotation_angle(source, flag=flag, tooth_position=tooth_position)
    source_rotated = ndimage.rotate(source, theta, reshape=True, cval=255)

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

    t = np.tan(phi)
    left_padding, right_padding = 0, 1
    if theta > 0:
        left_padding = round((y2 - opposite) * t)
        right_padding = round((adjacent - y1) * t)
    elif theta < 0:
        left_padding = round((adjacent - y1) * t)
        right_padding = round((y2 - opposite) * t)
    source_roi = source_roi[:, left_padding:-right_padding]

    if flag == 'upper':
        gum_sep_line += 30
    else:
        gum_sep_line -= 30

    return source_roi, gum_sep_line, theta, [left_padding, right_padding]


def quick_get_tooth(image_path, tooth_number):
    im = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)

    json_path = image_path.with_suffix('.json')

    target_list = [''.join(i) for i in zip('11223344', '37' * 4)]
    with open(json_path, 'r') as f:
        labels = json.load(f)
        labels.pop('imageData', None)

    teeth_label = list(filter(lambda item: item['label'] in target_list, labels['shapes']))

    for tooth_label in teeth_label:
        if tooth_label['label'] == tooth_number:
            points = tooth_label['points']
            points = list(map(lambda point: list(map(int, point)), points))

            [x1, y1], [x2, y2] = points
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            tooth_im = im[y1:y2, x1:x2]
            xyxy = np.array([x1, y1, x2, y2])

            return tooth_im, xyxy

    raise AttributeError(f'tooth number {tooth_number} not found.')
