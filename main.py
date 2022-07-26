import cv2
import torch
from glob import glob
from matplotlib import pyplot as plt
import matplotlib

from YOLO.util import get_teeth_ROI
from edge.util import tooth_isolation
from util import bounding_teeth_on_origin

from yolov5.utils.plots import Annotator, colors

if __name__ == '__main__':
    model = torch.hub.load(r'.\YOLO', 'custom', path=r'.\YOLO\weights\8-bound.pt', source='local')
    # Image
    images = glob('../Datasets/phase-2/*.jpg')

    results = model(images[:3])

    teeth_region = bounding_teeth_on_origin(results)
