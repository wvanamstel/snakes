import numpy as np
import os
import torch

from glob import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "crowdai", "snakes")
SEGMENTATION_MODEL_PATH = os.path.join(os.environ["DATA_PATH"], "aicrowd/snakes/segm_model")

SEG_MODEL = torch.load(os.path.join(SEGMENTATION_MODEL_PATH, "snake_seg.pt"))

with open(os.path.join(DATA_PATH, "class_idx_mapping.csv"), "r") as f_in:
    classes = [cl.split(',') for cl in f_in.readlines()[1:]]
    label_to_id = {_cl[1].rstrip(): _cl[0] for _cl in classes}

train_samples = glob(os.path.join(DATA_PATH, "train") + "/*/*", recursive=True)


class SnakeData(Dataset):
    def __init__(self, file_names, transformation=None):
        self.file_names = file_names
        self.transformation = transformation
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.file_names)

    def __get_item__(self, idx):
        image_file = self.file_names[idx]
        image = Image.open(image_file).convert("RGB")
        _id = image_file.spli("/")[-2].split("-")[1]
        if self.transformation:
            self.transformation(image)
        return image, _id


