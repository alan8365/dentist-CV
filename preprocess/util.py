import os
import json

import pandas as pd
import torch

from glob import glob


def rect_include_another(box1, box2, eps=1e-9):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

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
            labels = {shape['label'] for shape in shapes}

            filename, _ = os.path.splitext(os.path.basename(json_file))

            d.update({filename: {i: (i in labels) for i in col_names}})

    df = pd.DataFrame.from_dict(d, orient='index')
    if save:
        df.to_csv('label_TF.csv', index=True, index_label='filename')

    return df


def get_image_by_labels(target_labels, file_name='label_TF.csv'):
    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, index_col='filename')
    else:
        # FIXME path rewrite
        df = get_image_label()

    result_mask = df[target_labels].any(axis=1)

    return df[result_mask]


def get_labels_by_image(file_path, target_labels=None):
    with open(file_path, 'r', encoding='utf-8') as f:
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


# TODO classified by tooth number
if __name__ == '__main__':
    pass
