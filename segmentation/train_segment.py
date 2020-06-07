import glob
import os
import torch
import torchvision
import xml

from xml.etree import cElementTree as ElementTree
from parse_xml import XmlDictConfig
from PIL import Image
from torchvision import transforms as T

DATA_PATH = os.environ["LOCAL_DATA_PATH"]
MASK_PATH = os.path.join(DATA_PATH, 'aicrowd', 'snakes', 'anno')

def transform_image(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # add training data transforms
        pass
    return T.Compose(transforms)


class SnakeSegmentationDataset(object):
    def __init__(self, transforms=None):
        self.transforms = transforms
        mask_file_names = glob.glob(os.path.join(MASK_PATH, '*.xml'))
        mask_data = [XmlDictConfig(ElementTree.parse(f).getroot()) for f in mask_file_names]
        self.image_file_names = [mask.get("path") for mask in mask_data]
        self.bbox_data = [mask.get("object").get("bndbox") for mask in mask_data]

    def __getitem__(self, idx):
        image = Image.open(self.image_file_names[idx]).convert('RGB')
        bnd_box = [
            int(self.bbox_data[idx].get('xmin')),
            int(self.bbox_data[idx].get('ymin')),
            int(self.bbox_data[idx].get('xmax')),
            int(self.bbox_data[idx].get('ymax')),
            ]
        box = torch.as_tensor(bnd_box, dtype=torch.float32)
        label = torch.ones((1,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (box[2] - box[0]) * (box[3] - box[1])

        target = {}
        target["boxes"] = box
        target["labels"] = label
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_file_names)

dataset = SnakeSegmentationDataset(transform_image(train=True))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
# data = next(iter(data_loader))
