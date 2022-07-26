import cv2
from matplotlib import pyplot as plt
import matplotlib

from YOLO.util import get_teeth_ROI
from edge.util import tooth_isolation

from yolov5.utils.plots import Annotator, colors


def bounding_teeth_on_origin(results, save=False):
    teeth_roi = get_teeth_ROI(results)

    split_teeth = teeth_roi['split_teeth']
    teeth_roi = teeth_roi['images']

    tooth_position_dict = {
        0: 'left',
        1: 'middle',
        2: 'right'
    }

    teeth_region = {}
    matplotlib.use('module://backend_interagg')
    for file_name, data in teeth_roi.items():
        temp_region = {}
        for datum in data:
            flag = datum['flag']
            number = datum['number']
            im = datum['image']
            offset = datum['offset']

            tooth_position = tooth_position_dict[number]
            im_g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

            region = tooth_isolation(im_g, flag=flag, file_name=file_name, tooth_position=tooth_position, save=save)
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
        im0 = cv2.imread(f'../Datasets/phase-2/{file_name}.jpg')
        annotator = Annotator(im0, line_width=3, example=file_name)
        for tooth_number, body in data.items():
            xyxy = body['xyxy']
            # rotation_offset = body['rotation_offset']

            # adjust by rotation offset
            # xyxy[1] += rotation_offset[0]
            # xyxy[3] += rotation_offset[1]

            annotator.box_label(xyxy, str(tooth_number), color=(255, 0, 0))

        im1 = annotator.result()

        plt.imshow(im1)
        plt.title(file_name)
        plt.show()

    return teeth_region
