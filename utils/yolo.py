import torch

from yolov5.utils.plots import Annotator, colors, save_one_box
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def crop_by_two_tooth(left, right, margin=50):
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    x_items = torch.Tensor([left[2], right[0]])
    y_items = torch.Tensor([left[1], left[3], right[1], right[3]])

    crop = torch.Tensor([
        torch.min(x_items) - margin,
        torch.min(y_items),
        torch.max(x_items) + margin,
        torch.max(y_items) + margin,
    ])
    crop = crop.int()

    return crop


def get_teeth_ROI(detected_results, save=False):
    flag_dict = {
        'upper': [
            '17', '13', '23', '27'
        ],
        'lower': [
            '47', '43', '33', '37'
        ]
    }

    tooth_position_dict = {
        0: 'left',
        1: 'middle',
        2: 'right'
    }

    images = {}
    split_teeth = {}
    for i in range(len(detected_results)):
        file_name = detected_results.files[i][:-4]
        bounds = detected_results.xyxy[i]
        img = detected_results.imgs[i]

        images[file_name] = []
        split_teeth[file_name] = {}
        for flag in ('upper', 'lower'):
            teeth_dict = {}
            flag_list = flag_dict[flag]

            for j in range(len(bounds)):
                *xyxy, _, cls = bounds[j]

                cls = int(cls.item())
                name = detected_results.names[cls]
                teeth_dict[name] = xyxy
                split_teeth[file_name][name] = {'xyxy': xyxy}

            teeth_detected_flag = [f in teeth_dict for f in flag_list]

            if sum(teeth_detected_flag) < 2:
                # print(file_name + ':' + str(teeth_dict.keys()))
                continue

            tooth_detected_tuples = [(flag_list[j], flag_list[j + 1], j) for j in range(len(flag_list) - 1) if
                                     teeth_detected_flag[j] and teeth_detected_flag[j + 1]]

            for left_tooth_number, right_tooth_number, number in tooth_detected_tuples:
                left_tooth = teeth_dict[left_tooth_number]
                right_tooth = teeth_dict[right_tooth_number]

                region = crop_by_two_tooth(left_tooth, right_tooth)
                tooth_position = tooth_position_dict[number]

                save_filename = f'{flag}-{number}-{file_name}'
                save_file = Path(f'./crops/{save_filename}.jpg')

                im = save_one_box(region, img, save=save, file=save_file)
                # print(f'First ROI process: {save_filename} done.')

                image_data = {
                    'flag': flag,
                    'number': number,
                    'tooth_position': tooth_position,
                    'org_file_name': file_name,
                    'offset': np.array([region[0], region[1]]),
                    'image': im,
                }

                images[file_name].append(image_data)

    return {'images': images, 'split_teeth': split_teeth}

    # matplotlib.use('module://backend_interagg')
    #
    # fig, axs = plt.subplots(1, 3)
    #
    # axs[0].imshow(im1)
    # axs[1].imshow(im2)
    # axs[2].imshow(im3)
    #
    # plt.show()


if __name__ == "__main__":
    model = torch.hub.load('.', 'custom', path=r'.\weights\8-bound.pt', source='local')
    # Image
    imgs = glob('../../Datasets/phase-2/*.jpg')
    # Inference
    results = model(imgs[:10])

    get_teeth_ROI(results)
