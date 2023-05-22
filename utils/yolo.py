from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from yolov5.utils.general import increment_path
from yolov5.utils.plots import save_one_box


def test_curve_factory(middle_width, middle_height, a=-0.0004):
    def foo(x):
        return a * (x - middle_width) ** 2 + middle_height

    return foo


def crop_by_two_tooth(left_tooth, right_tooth, margin=50, padding=50):
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # x_items = torch.Tensor([left_tooth[2], right_tooth[0]])
    # y_items = torch.Tensor([left_tooth[1], left_tooth[3], right_tooth[1], right_tooth[3]])
    x_indices = [0, 2]
    y_indices = [1, 3]
    x_items = torch.Tensor([*left_tooth[x_indices], *right_tooth[x_indices]])
    y_items = torch.Tensor([*left_tooth[y_indices], *right_tooth[y_indices]])

    x_items_padded = [torch.sum(x_items[:2]) / 2, torch.sum(x_items[2:]) / 2]

    crop = torch.Tensor([
        x_items_padded[0],
        torch.min(y_items) - margin,
        x_items_padded[1],
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

        # TODO teeth position check
        im_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        tooth_bounds = []
        for j in range(len(bounds)):
            x1, y1, x2, y2, _, cls = bounds[j]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            xyxy = torch.vstack([x1, y1, x2, y2])

            cls = int(cls.item())
            name = detected_results.names[cls]
            tooth_bounds.append({'xyxy': xyxy, 'name': name})

        teeth_dict = {}
        for j in range(len(tooth_bounds)):
            xyxy = tooth_bounds[j]['xyxy']
            name = tooth_bounds[j]['name']

            teeth_dict[name] = xyxy

            crop_image = crop_by_xyxy(img, xyxy.int())
            split_teeth[file_name][name] = {'xyxy': xyxy, 'crop_image': crop_image}

        tooth_detected_tuples = []
        for flag in ('upper', 'lower'):
            flag_list = flag_dict[flag]
            teeth_detected_flag = [f in teeth_dict for f in flag_list]
            tooth_detected_tuples += [
                (flag_list[j], flag_list[j + 1], j, flag)
                for j in range(len(flag_list) - 1)
                if teeth_detected_flag[j] and teeth_detected_flag[j + 1]
            ]

        for left_tooth_number, right_tooth_number, number, flag in tooth_detected_tuples:
            left_tooth = teeth_dict[left_tooth_number]
            right_tooth = teeth_dict[right_tooth_number]

            region = crop_by_two_tooth(left_tooth, right_tooth)
            tooth_position = tooth_position_dict[number]

            save_filename = f'{flag}-{number}-{file_name}'
            save_file = Path(f'./crops/{save_filename}.jpg')

            x1, y1, x2, y2 = region
            im = img[y1:y2, x1:x2]
            # print(f'First ROI process: {save_filename} done.')

            left_padding = torch.div(left_tooth[2] - left_tooth[0], 2, rounding_mode='floor')
            right_padding = torch.div(right_tooth[2] - right_tooth[0], 2, rounding_mode='floor')
            image_data = {
                'flag': flag,
                'tooth_position': tooth_position,
                'org_file_name': file_name,
                'offset': np.array([region[0], region[1]]),
                'padding': {'left': left_padding, 'right': right_padding},
                'image': im,
                'xyxy': region,
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


def crop_by_xyxy(image, xyxy, save=False, file=Path('im.jpg')):
    x1, y1, x2, y2 = xyxy
    result = image[y1:y2, x1:x2]

    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(result).save(f, quality=95, subsampling=0)
    return result
