#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import torch
from glob import glob
from scipy import ndimage
import math
import json
import numpy as np

import cv2
from matplotlib import pyplot as plt
import matplotlib

from pathlib import Path
from math import sin, cos
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

# # One file check

# In[5]:


image_labels = {}
labels = get_labels_by_image(filepath_json, target_labels)
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

# In[6]:


image_labels = {}
for filename in image_labels_df.index:
    print(filename)
    filepath_image = data_dir / f'{filename}.jpg'
    filepath_json = data_dir / f'{filename}.json'

    results = model(filepath_image)

    teeth_roi = get_teeth_ROI(results)
    teeth_roi_images = teeth_roi['images'][filename]
    teeth_roi_split_teeth = teeth_roi['split_teeth']

    labels = get_labels_by_image(filepath_json, target_labels)
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
