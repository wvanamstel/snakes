import numpy as np
import os
import torch

from util import collate_fn

from glob import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "aicrowd", "snakes")

SEGMENTATION_MODEL_PATH = os.path.join(os.environ["DATA_PATH"], "aicrowd/snakes/segm_model", "snake_seg.pt")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# SEG_MODEL = load_segmentation_model(SEGMENTATION_MODEL_PATH)


# class SnakeData(Dataset):
    # def __init__(self, file_names, transformation=None):
        # self.file_names = file_names
        # self.transformation = transformation
        # self.label_to_id = label_to_id

    # def __len__(self):
        # return len(self.file_names)

    # def __get_item__(self, idx):
        # image_file = self.file_names[idx]
        # image = Image.open(image_file).convert("RGB")
        # _id = image_file.split("/")[-2].split("-")[1]
        # if self.transformation:
            # self.transformation(image)
        # return image, _id

def clf_image_transform(train: bool):
    transforms = []
    # transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_segmentation_model(model_path: str):
    seg_model = torch.load(model_path)
    seg_model.to(DEVICE)
    seg_model.eval()
    return seg_model


class SnakeClfData(Dataset):
    def __init__(self, file_names, label_to_id, transformation=None, segmentation_model_path=None):
        self.file_names = file_names
        self.transformation = transformation
        self.label_to_id = label_to_id
        if segmentation_model_path:
            self.seg_model = load_segmentation_model(segmentation_model_path)
        else:
            self.seg_model = False

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_file = self.file_names[idx]
        image = Image.open(image_file).convert("RGB")
        image = T.ToTensor()(image).unsqueeze(0)
        image = image.to(DEVICE)
        image_id = image_file.split("/")[-2].split("-")[1]
        image_id = self.label_to_id[image_id]
        if self.seg_model:
            segmentation = self.seg_model(image)
            mask = segmentation[0]['boxes'].to(torch.int)[0]
            mask_height = mask[3] - mask[1]
            mask_width = mask[2] - mask[0]
            mask_top = mask[1]
            mask_left = mask[0]
            image = T.functional.crop(image, mask_top, mask_left, mask_height, mask_width)
        if self.transformation is not None:
            self.transformation(image)
        return image, image_id


with open(os.path.join(DATA_PATH, "class_idx_mapping.csv"), "r") as f_in:
    classes = [cl.split(',') for cl in f_in.readlines()[1:]]
    label_to_snake_name = {_cl[1].rstrip(): _cl[0] for _cl in classes}
    label_to_idx = {label: i for i, label in enumerate(label_to_snake_name)}

train_samples = glob(os.path.join(DATA_PATH, "train") + "/*/*", recursive=True)

dataset = SnakeClfData(train_samples, label_to_idx, clf_image_transform(train=True), segmentation_model_path=SEGMENTATION_MODEL_PATH)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


