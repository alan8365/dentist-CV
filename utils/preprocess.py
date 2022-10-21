import os
import json

import torch
import numpy as np
import pandas as pd

from glob import glob
from math import sin, cos


def xyxy_reformat(x):
    # Convert nx4 boxes to xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = x[:, [0, 2]].min(axis=1)  # right point x
    y[:, 1] = x[:, [1, 3]].min(axis=1)  # top point y
    y[:, 2] = x[:, [0, 2]].max(axis=1)  # left point x
    y[:, 3] = x[:, [1, 3]].max(axis=1)  # bottom point y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def rect_center_distance(box1, box2, xyxy=True):
    box1 = torch.Tensor(box1)
    box2 = torch.Tensor(box2)

    xyxy2xywh([box1, box2])


def rect_include_another(box_out, box_in, eps=1e-9):
    # Get the coordinates of bounding boxes
    box_in = torch.Tensor(box_in)
    box_out = torch.Tensor(box_out)

    b1_x1, b1_y1, b1_x2, b1_y2 = box_out[0], box_out[1], box_out[2], box_out[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box_in[0], box_in[1], box_in[2], box_in[3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    # w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    # union = w1 * h1 + w2 * h2 - inter + eps
    union = w2 * h2 + eps

    iou = inter / union
    return iou  # IoU


def get_image_label(dataset_dir=None, save=True):
    # FIXME path
    if not dataset_dir:
        dataset_dir = os.path.join('..', '..', 'Datasets', 'phase-2')
    json_files = glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)

    col_names = ['13', '17', '23', '27', '33', '37', '43', '47',
                 'Imp', 'R.R', 'bridge', 'caries', 'crown', 'embedded',
                 'endo', 'filling', 'impacted', 'post']

    d = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            shapes = data['shapes']
            labels = [shape['label'] for shape in shapes]

            filename, _ = os.path.splitext(os.path.basename(json_file))

            d.update({filename: {i: (labels.count(i)) for i in col_names}})

    df = pd.DataFrame.from_dict(d, orient='index')
    if save:
        df.to_csv('label_TF.csv', index=True, index_label='filename')

    return df


def get_image_by_labels(target_labels=None, file_name='label_TF.csv'):
    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, index_col='filename')
    else:
        # FIXME path rewrite
        df = get_image_label()

    if target_labels:
        result_mask = df[target_labels].any(axis=1)
        df = df[result_mask]

    return df


def get_labels_by_image(filepath, target_labels=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        shapes = data['shapes']

        if target_labels:
            result = [shape for shape in shapes if shape['label'] in target_labels]
        else:
            result = [shape for shape in shapes]

    return result


def get_classification_format(teeth_region, target_labels, dataset_dir=None):
    """

    :param teeth_region: output from bounding_teeth_on_origin
    :param target_labels:
    """
    result = []
    if not dataset_dir:
        dataset_dir = os.path.join('..', '..', 'Datasets', 'phase-2')
    image_labels_df = get_image_by_labels(target_labels)[target_labels]

    # Image
    images = list(map(lambda s: os.path.join(dataset_dir, f'{s}.jpg'), image_labels_df.index))
    test_image_number = 3

    jsons = list(teeth_region.keys())

    for json_filename in jsons:
        json_filepath = os.path.join(dataset_dir, f'{json_filename}.json')
        labels_list = get_labels_by_image(json_filepath, target_labels)
        for labels_data in labels_list:
            label = labels_data['label']
            points = labels_data['points']
            xyxy_in = torch.Tensor([points[0][0], points[0][1], points[1][0], points[1][1]])

            for key, value in teeth_region[json_filename].items():
                xyxy_out = value['xyxy']
                if rect_include_another(xyxy_out, xyxy_in) > 0.5:
                    result.append({f'{json_filename}-{key}': label})


def recovery_rotated_bounding(theta, org_image_shape, bounding_boxes):
    rot_matrix = np.copy([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    in_center, out_center = get_rotation_center(org_image_shape, rot_matrix)
    offset = in_center - out_center

    # Compute rotation
    xywh_rot_matrix = np.array([[cos(theta), -sin(theta), 0, 0],
                                [sin(theta), cos(theta), 0, 0],
                                [0, 0, cos(theta), sin(abs(theta))],
                                [0, 0, sin(abs(theta)), cos(theta)], ])
    bounding_boxes_rotated = np.array(bounding_boxes)
    bounding_boxes_rotated = xyxy2xywh(bounding_boxes_rotated)
    bounding_boxes_rotated = bounding_boxes_rotated.transpose()

    bounding_boxes_rotated = xywh_rot_matrix @ bounding_boxes_rotated

    margin = np.array([offset[0], offset[1], 0, 0])
    bounding_boxes_rotated = bounding_boxes_rotated + margin[:, None]
    bounding_boxes_rotated = xywh2xyxy(bounding_boxes_rotated.transpose()).astype(int)

    return bounding_boxes_rotated


def get_rotation_center(org_image_shape, rot_matrix):
    in_plane_shape = np.array(org_image_shape)
    # Compute transformed input bounds
    iy, ix = in_plane_shape
    out_bounds = rot_matrix @ [[0, 0, iy, iy],
                               [0, ix, 0, ix]]
    # Compute the shape of the transformed input plane
    out_plane_shape = (out_bounds.ptp(axis=1) + 0.5).astype(int)
    out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
    in_center = (in_plane_shape - 1) / 2
    return in_center, out_center


def rotate_bounding_boxes(theta, org_image_shape, bounding_boxes):
    rot_matrix = np.copy([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    in_center, out_center = get_rotation_center(org_image_shape, rot_matrix)
    offset = out_center - in_center
    offset = np.tile(offset, 2)

    rot_matrix = np.copy([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])
    bounding_boxes_rotated = np.array(bounding_boxes)
    boxes_shape = bounding_boxes_rotated.shape

    bounding_boxes_rotated = bounding_boxes_rotated + np.tile(offset, boxes_shape[0])
    bounding_boxes_rotated = bounding_boxes_rotated.reshape((boxes_shape[0], 2, 2), order='F')
    bounding_boxes_rotated = (rot_matrix @ bounding_boxes_rotated).reshape((boxes_shape[0], 4), order='F')

    return bounding_boxes_rotated


# TODO classified by tooth number
if __name__ == '__main__':
    phi = 0.3490658503988659
    shape = [326, 443]
    xyxy = [[144.91, 34.606, 276.73, 136.12]]

    xyxy = rotate_bounding_boxes(phi, shape, xyxy)
