import glob
import os
import torch
import torchvision
import xml
import utils

from engine import train_one_epoch, evaluate
from xml.etree import cElementTree as ElementTree
from parse_xml import XmlDictConfig
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
DATA_PATH = os.environ["DATA_PATH"]
MASK_PATH = os.path.join(DATA_PATH, 'aicrowd', 'snakes', 'anno')


def transform_image(train):
    transforms = []
    transforms.append(T.ToTensor())
    # if train:
        # transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class SnakeSegmentationDataset(object):
    def __init__(self, transforms=None):
        self.transforms = transforms
        mask_file_names = glob.glob(os.path.join(MASK_PATH, '*.xml'))
        mask_data = [XmlDictConfig(ElementTree.parse(f).getroot()) for f in mask_file_names]
        self.image_file_names = [DATA_PATH + mask.get("path").split('data')[1] for mask in mask_data]
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
        area = (box[2] - box[0]) * (box[3] - box[1])
        box = box.unsqueeze(0)
        label = torch.ones((1,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = box
        target["labels"] = label
        target["image_id"] = image_id
        target["area"] = area.unsqueeze(0)
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        return image.squeeze(0), target

    def __len__(self):
        return len(self.image_file_names)

def get_segmentation_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.seed()

    num_classes = 2
    dataset = SnakeSegmentationDataset(transform_image(train=True))
    dataset_test = SnakeSegmentationDataset(transform_image(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    print(indices)
    num_examples = len(indices)
    dataset = torch.utils.data.Subset(dataset, indices[:-int(num_examples*0.2)])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-int(num_examples*0.2):])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    model = get_segmentation_model(num_classes)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 2

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

main()

