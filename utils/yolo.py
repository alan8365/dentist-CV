import torch
from PIL import Image
from yolov5.utils.general import increment_path

from yolov5.utils.plots import Annotator, colors, save_one_box
from pathlib import Path

import numpy as np
import cv2

from utils.general import integral_intensity_projection


def test_curve_factory(middle_width, middle_height, a=-0.0004):
    def foo(x):
        return a * (x - middle_width) ** 2 + middle_height

    return foo


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

    tooth_number_flag_dict = {
        '1': ('upper', 'left'),
        '2': ('upper', 'right'),
        '3': ('lower', 'right'),
        '4': ('lower', 'left'),
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
        height, width = im_g.shape
        middle_height, middle_width = height // 2, width // 2

        hor, _ = integral_intensity_projection(im_g)

        tooth_bounds = []
        for j in range(len(bounds)):
            *xyxy, _, cls = bounds[j]
            xyxy = torch.vstack(xyxy)

            cls = int(cls.item())
            name = detected_results.names[cls]
            tooth_bounds.append({'xyxy': xyxy, 'name': name})

        for flag in ('upper', 'lower'):
            teeth_dict = {}
            flag_list = flag_dict[flag]

            for j in range(len(tooth_bounds)):
                xyxy = tooth_bounds[j]['xyxy']
                name = tooth_bounds[j]['name']

                teeth_dict[name] = xyxy

                crop_image = crop_by_xyxy(img, xyxy.int())
                split_teeth[file_name][name] = {'xyxy': xyxy, 'crop_image': crop_image, 'is_missing': False}

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
