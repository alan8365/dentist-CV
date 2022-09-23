#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from scipy import ndimage
import math
import json
import torch
import numpy as np

import cv2
import matplotlib

from matplotlib import pyplot as plt

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from utils.yolo import get_teeth_ROI
from utils.edge import tooth_isolation, gum_jaw_separation, vertical_separation, bounding_teeth_on_origin
from utils.preprocess import recovery_rotated_bounding, xyxy2xywh, get_image_by_labels, get_labels_by_image
from utils.preprocess import xyxy_reformat, rotate_bounding_boxes, rect_include_another

matplotlib.use('module://matplotlib_inline.backend_inline')
load_dotenv()

# In[2]:


target_labels = ['caries', 'endo', 'post', 'crown']
image_labels_df = get_image_by_labels(target_labels)[target_labels]

image_labels_df

# In[3]:


data_dir = Path('..') / '..' / 'Datasets' / 'phase-2'

filename = image_labels_df.index[1]
filepath_image = data_dir / f'{filename}.jpg'
filepath_json = data_dir / f'{filename}.json'

# In[4]:


tooth_position_dict = {
    0: 'left',
    1: 'middle',
    2: 'right'
}

model = torch.hub.load(r'..\YOLO', 'custom', path=r'..\YOLO\weights\8-bound.pt', source='local')
# Image
# Inference
results = model(filepath_image)

teeth_roi = get_teeth_ROI(results)
teeth_roi_images = teeth_roi['images'][filename]
teeth_roi_split_teeth = teeth_roi['split_teeth']
teeth_roi_images

# In[5]:


teeth_roi = get_teeth_ROI(results)
teeth_roi_images = teeth_roi['images'][filename]
teeth_roi_split_teeth = teeth_roi['split_teeth']
teeth_roi_images

#

# # One file check

# In[6]:


image_labels = {}
labels = get_labels_by_image(filepath_json, target_labels)
for target_roi in teeth_roi_images:
    flag = target_roi['flag']
    target_roi_image = target_roi['image']
    tooth_position = tooth_position_dict[target_roi['number']]
    im_g = cv2.cvtColor(target_roi_image, cv2.COLOR_RGBA2GRAY)
    im_g_shape = np.array(np.array(im_g.shape)[[1, 0]])

    isolation_data = tooth_isolation(im_g, flag=flag, tooth_position=tooth_position, rotation_fix=False)
    if not isolation_data:
        continue
    regions = isolation_data['crop_regions']
    theta = isolation_data['angle']
    offset = target_roi['offset']

    phi = math.radians(theta)
    for label in labels:
        xyxy = np.hstack(label['points'])  # [x, y, x, y]
        xyxy = xyxy_reformat(np.array([xyxy]))

        xyxy = xyxy - np.tile(offset, 2)
        if xyxy.min() < 0:
            continue

        xyxy = rotate_bounding_boxes(phi, im_g_shape, xyxy)
        xyxy = xyxy[0].astype(int)

        for tooth_number, region in regions.items():
            tooth_xyxy = region['xyxy']
            if rect_include_another(tooth_xyxy, xyxy) > 0.5:
                key = f'{filename}-{tooth_number}'
                if not key in image_labels.keys():
                    image_labels[key] = []
                image_labels[key].append(label['label'])

image_labels
# {'00008026-11': ['post', 'endo', 'crown'],
#  '00008026-21': ['post', 'endo', 'crown'],
#  '00008026-12': ['crown'],
#  '00008026-22': ['crown'],
#  '00008026-26': ['crown'],
#  '00008026-46': ['caries'],
#  '00008026-32': ['crown'],
#  '00008026-36': ['crown', 'post', 'endo']}

# In[7]:


image_labels = {}


def method_name():
    global target_roi, target_roi_image, flag, tooth_position, im_g, im_g_shape, isolation_data, regions, theta, offset, phi, label, xyxy, tooth_number, region, tooth_xyxy, key
    for target_roi in teeth_roi_images:
        target_roi_image = target_roi['image']
        flag = target_roi['flag']
        tooth_position = tooth_position_dict[target_roi['number']]
        im_g = cv2.cvtColor(target_roi_image, cv2.COLOR_RGBA2GRAY)
        im_g_shape = np.array(np.array(im_g.shape)[[1, 0]])

        isolation_data = tooth_isolation(im_g, flag=flag, tooth_position=tooth_position, rotation_fix=False)
        if not isolation_data:
            continue
        regions = isolation_data['crop_regions']
        theta = isolation_data['angle']
        offset = target_roi['offset']

        phi = math.radians(theta)
        for label in labels:
            xyxy = np.hstack(label['points'])  # [x, y, x, y]
            xyxy = xyxy_reformat(np.array([xyxy]))

            xyxy = xyxy - np.tile(offset, 2)
            if xyxy.min() < 0:
                continue

            xyxy = rotate_bounding_boxes(phi, im_g_shape, xyxy)
            xyxy = xyxy[0].astype(int)

            for tooth_number, region in regions.items():
                tooth_xyxy = region['xyxy']
                if rect_include_another(tooth_xyxy, xyxy) > 0.5:
                    key = f'{filename}-{tooth_number}'
                    if not key in image_labels.keys():
                        image_labels[key] = []
                    image_labels[key].append(label['label'])


for filename in tqdm(image_labels_df.index):
    filepath_image = data_dir / f'{filename}.jpg'
    filepath_json = data_dir / f'{filename}.json'

    results = model(filepath_image)

    teeth_roi = get_teeth_ROI(results)
    teeth_roi_images = teeth_roi['images'][filename]
    teeth_roi_split_teeth = teeth_roi['split_teeth']

    labels = get_labels_by_image(filepath_json, target_labels)
    method_name()

# In[8]:


# for tooth_number, r in region.items():
#     print(r)
# target_roi
# label
# !jupyter nbconvert --to script prepare_classify_format.ipynb

j = json.dumps(image_labels)

with open('image_labels_for_classify.json.bak', 'w') as f:
    f.write(j)

# ## Check json file and jpg file pair

# In[9]:


jpgs = list(data_dir.glob('*.jpg'))
jsons = list(data_dir.glob('*.json'))

jpgs_set = {jpg.stem for jpg in jpgs}
jsons_set = {json_file.stem for json_file in jsons}

(jpgs_set - jsons_set) | (jsons_set - jpgs_set)
