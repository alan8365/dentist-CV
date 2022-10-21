# %%
import cv2
import matplotlib
import matplotlib.pyplot as plt
import timm
import torch
import numpy as np

from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from timm.data.loader import create_loader
from timm.data.dataset import ImageDataset
import torchvision.transforms as transforms

import os

from ViT.tooth_crop_dataset import ToothCropClassDataset
from utils.preprocess import rect_include_another, rotate_bounding_boxes, xyxy_reformat, xyxy2xywh
from utils.yolo import get_teeth_ROI
from utils.edge import tooth_isolation, bounding_teeth_on_origin, get_all_teeth

import warnings

warnings.filterwarnings("ignore")


def main(dir):
    load_dotenv()
    YOLO_model_dir = Path(os.getenv('YOLO_MODEL_DIR'))

    tooth_detect_model = torch.hub.load('YOLO', 'custom', path=YOLO_model_dir / '8-bound.pt', source='local',
                                        verbose=False)
    anomaly_detect_model = torch.hub.load('YOLO', 'custom', path=YOLO_model_dir / 'anomaly.pt', source='local')

    data_dir = Path(dir)
    image_names = list(data_dir.glob('*.jpg'))

    results = tooth_detect_model(image_names)
    anomaly_results = anomaly_detect_model(image_names)

    teeth_region = get_all_teeth(results)
    teeth_roi = get_teeth_ROI(results)

    # Save tooth crop image
    temp_dir = '.' / Path(os.getenv('TEMP_DIR')) / 'crop_tooth_image'

    missing_tooth = []
    for file in temp_dir.glob('*.jpg'):
        os.remove(file)

    for filename, region in teeth_region.items():
        for region_name, tooth_region in region.items():
            for tooth_number, data in tooth_region['crop_regions'].items():
                crop_image = data['crop_image']
                is_missing = data['is_missing']

                if is_missing:
                    missing_tooth.append((filename, tooth_number))
                    continue
                save_filepath = temp_dir / f'{filename} {region_name} {tooth_number}.jpg'

                temp_im = Image.fromarray(crop_image)
                temp_im.save(save_filepath)

    # Vit model loading
    model_dir = '.' / Path(os.getenv('ViT_MODEL_DIR'))
    model_path = model_dir / 'classifier-6.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vit_model = timm.create_model('vit_base_patch16_224', num_classes=6)
    vit_model.load_state_dict(torch.load(model_path, map_location=device))

    batch_size = 16
    num_workers = 0

    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        # (lambda image: padding_to_size(image, 224)),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(mean=0.5, std=0.5),
    ])
    target_transform = transforms.Compose([
        (lambda y: torch.Tensor(y)),
    ])
    dataset = ImageDataset(temp_dir, transform=transform)

    if torch.cuda.is_available():
        dataloader = create_loader(dataset, (3, 224, 224), 4)
    else:
        dataloader = create_loader(dataset, (3, 224, 224), 4, use_prefetcher=False)

    size = len(dataloader.dataset)

    vit_model.to(device)
    vit_model.eval()

    threshold = torch.Tensor([0.5, 0.85, 0.5, 0.5, 0.5, 0.5]).to(device)
    pred_encodes = []
    # target_labels = ['caries', 'endo', 'post', 'crown']
    target_labels = ['R.R', 'caries', 'crown', 'endo', 'filling', 'post']
    with torch.no_grad():
        for batch, (X, _) in enumerate(dataloader):
            X = X.to(device)

            # Compute prediction error
            pred = vit_model(X)
            pred_encode = pred > threshold
            pred_encodes.append(pred_encode.cpu().numpy())

    pred_encodes = np.vstack(pred_encodes)
    detected_list = [()] * len(pred_encodes)
    for i, pred_encode in enumerate(pred_encodes):
        detected_list[i] = tuple((target_labels[j] for j, checker in enumerate(pred_encode) if checker))

    # Final process
    tooth_anomaly_dict = {anomaly_results.files[i][:-4]: {} for i in range(len(anomaly_results))}

    for i, detected in enumerate(detected_list):
        current_filename, region_name, tooth_number = dataset.filename(i).split()
        tooth_number = int(tooth_number[:2])

        if tooth_number < 50:
            tooth_anomaly_dict[current_filename][tooth_number] = set(detected)

    iou_threshold = 0.5
    region_wisdom_tooth_dict = {
        'upper-left': 18,
        'upper-right': 28,
        'lower-left': 48,
        'lower-right': 48,
    }

    for i in range(len(anomaly_results)):
        current_filename = anomaly_results.files[i][:-4]
        bounds = anomaly_results.xyxy[i]
        for j in range(len(bounds)):
            *xyxy, _, cls = bounds[j]
            xyxy = list(map(lambda t: t.cpu(), xyxy))

            cls = int(cls.item())
            name = anomaly_results.names[cls]
            if name in target_labels:
                continue

            # for x8 tooth
            if name in ['embedded', 'impacted']:
                min_distance = np.inf
                near_region = ''
                for region_data in teeth_roi['images'][current_filename]:
                    region_xyxy = region_data['xyxy']
                    region_xywh = xyxy2xywh(np.vstack([region_xyxy]))[0]

                    xywh = xyxy2xywh(np.vstack([xyxy]))[0]

                    distance = np.linalg.norm(xywh[:2] - region_xywh[:2])
                    if distance < min_distance:
                        min_distance = distance
                        near_region = f'{region_data["flag"]}-{region_data["tooth_position"]}'
                if near_region in region_wisdom_tooth_dict:
                    tooth_number = region_wisdom_tooth_dict[near_region]
                    if tooth_number not in tooth_anomaly_dict[current_filename].keys():
                        tooth_anomaly_dict[current_filename][tooth_number] = {name}
                    else:
                        tooth_anomaly_dict[current_filename][tooth_number].add(name)
                continue

            # for normal tooth
            located_regions = {}
            for region_data in teeth_roi['images'][current_filename]:
                region_xyxy = region_data['xyxy']
                if rect_include_another(region_xyxy, xyxy) > iou_threshold:
                    located_regions[f'{region_data["flag"]}-{region_data["tooth_position"]}'] = region_data

            for located_region, region_data in located_regions.items():
                region_tooth_data = teeth_region[current_filename][located_region]
                tooth_angle = np.radians(region_tooth_data['angle'])

                offset = region_data['offset']
                region_image_shape = np.array(np.array(region_data['image'].shape)[[1, 0]])

                rotated_xyxy = [xyxy]
                rotated_xyxy = np.array(rotated_xyxy) - np.tile(offset, 2)
                rotated_xyxy = rotate_bounding_boxes(tooth_angle, region_image_shape, rotated_xyxy)
                rotated_xyxy = rotated_xyxy[0].astype(int)
                for tooth_number, tooth_data in region_tooth_data['crop_regions'].items():
                    tooth_xyxy = tooth_data['xyxy']

                    if rect_include_another(tooth_xyxy, rotated_xyxy) > iou_threshold:
                        # if tooth_number not in result[current_filename].keys():
                        #     result[current_filename][tooth_number] = {name}
                        # else:
                        tooth_anomaly_dict[current_filename][tooth_number].add(name)

    for pair in missing_tooth:
        filename, tooth_number = pair
        tooth_anomaly_dict[filename][tooth_number] = {'missing'}

    return tooth_anomaly_dict
