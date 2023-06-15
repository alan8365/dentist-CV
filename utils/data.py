import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator


class DentexDataset:
    """A class for loading and processing dental x-ray images.

    Args:
        root_dir: The path to the root directory of the dataset.
        train: Whether to load the training or validation set.
        task: The task to perform on the dataset.

    Raises:
        ValueError: If the `task` argument is not one of `quadrant`, `quadrant_enumeration`, or `quadrant_enumeration_disease`.
    """

    def __init__(self, root_dir: Path, train=True, task='quadrant_enumeration'):
        self.root_dir = root_dir
        self.train = train
        self.task = task

        if task not in ['quadrant', 'quadrant_enumeration', 'quadrant_enumeration_disease']:
            raise ValueError(f'Invalid task: {task}')

        # Load the JSON file containing the annotations.
        if train:
            task_root_dir = self.root_dir / 'training_data' / task
            with open(task_root_dir / f'train_{task}.json') as f:
                self.json_content = json.load(f)
        else:
            task_root_dir = self.root_dir / 'validation_data' / 'quadrant_enumeration_disease'
            self.json_content = None

        # Get the paths to the x-ray images.
        self.task_root_dir = task_root_dir
        self.img_paths = list((task_root_dir / 'xrays').glob('*.png'))
        self.img_names = [i.name for i in self.img_paths]

        # Create a dictionary that maps from image name to a dictionary of annotations.
        image_id_dict = {i['id']: i['file_name'] for i in self.json_content['images']}
        annotations = {}
        for annotation in self.json_content['annotations']:
            image_id = annotation['image_id']
            image_name = image_id_dict[image_id]
            bbox = self.xywh_to_xyxy(annotation['bbox'])
            name = f'{annotation["category_id_1"] + 1}{annotation["category_id_2"] + 1}'

            if image_name not in annotations.keys():
                annotations[image_name] = {}

            annotations[image_name][name] = np.array(bbox)

        self.annotations = annotations

    def __len__(self):
        """Returns the number of x-ray images in the dataset."""
        return len(self.img_names)

    def __getitem__(self, idx):
        """Returns the x-ray image and annotations at the specified index."""
        img_name = self.img_names[idx]
        img_path = self.img_paths[idx]

        img = cv2.imread(str(img_path))
        annotations = self.annotations[img_name]
        return img, annotations

    def plot(self, image_name):
        """Plots the x-ray image with the annotations.

        Args:
          image_name: The name of the x-ray image.

        """
        # Read the x-ray image.
        # `cv2.imread()` reads an image from a file and returns a NumPy array.
        im = cv2.imread(str(self.task_root_dir / 'xrays' / image_name))

        # Create an annotator object.
        # `Annotator()` is a class that can be used to annotate x-ray images.
        annotator = Annotator(im, line_width=3, example=image_name)

        # Annotate the x-ray image with the tooth numbers.
        # `annotator.box_label()` draws a box around a tooth and labels it with the corresponding tooth number.
        for tooth_number, xyxy in self.annotations[image_name].items():
            annotator.box_label(xyxy, str(tooth_number), color=(255, 0, 0))

        # Plot the annotated x-ray image.
        # `plt.imshow()` plots an image.
        im1 = annotator.result()
        plt.imshow(im1)

        # Show the plot.
        plt.show()

    @staticmethod
    def xywh_to_xyxy(box):
        """Converts a bounding box from xywh format to xyxy format.

        Args:
            box: A list of four numbers, representing the bounding box in xywh format.

        Returns:
            A list of four numbers, representing the bounding box in xyxy format.
        """
        x, y, w, h = box
        return [x, y, x + w, y + h]
