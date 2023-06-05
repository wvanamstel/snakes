import glob
import os
import xml
from xml.etree import cElementTree as ElementTree

import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
from engine import evaluate, train_one_epoch
from parse_xml import XmlDictConfig

DATA_PATH = os.environ["DATA_PATH"]
MASK_PATH = os.path.join(DATA_PATH, "aicrowd", "snakes", "anno")
LOG_PATH = os.path.join(DATA_PATH, "aicrowd", "snakes", "logs")

WRITER = SummaryWriter(log_dir=LOG_PATH)


def transform_image(train):
    """transform_image.
    Transform images for segmentation model training

    :param train: (bool) True for training set, False otherwise
    """
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class SnakeSegmentationDataset(object):
    def __init__(self, transforms=None):
        self.transforms = transforms
        mask_file_names = glob.glob(os.path.join(MASK_PATH, "*.xml"))
        mask_data = [
            XmlDictConfig(ElementTree.parse(f).getroot()) for f in mask_file_names
        ]
        self.image_file_names = [
            DATA_PATH + mask.get("path").split("data")[1] for mask in mask_data
        ]
        self.bbox_data = [mask.get("object").get("bndbox") for mask in mask_data]

    def __getitem__(self, idx):
        image = Image.open(self.image_file_names[idx]).convert("RGB")
        bnd_box = [
            int(self.bbox_data[idx].get("xmin")),
            int(self.bbox_data[idx].get("ymin")),
            int(self.bbox_data[idx].get("xmax")),
            int(self.bbox_data[idx].get("ymax")),
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
    """get_segmentation_model.

    :param num_classes: (int) number of classes (2 for the bounding box, snake/no snake)
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 2
    batch_size = 3
    # Set up data loaders for train/test split
    dataset = SnakeSegmentationDataset(transform_image(train=True))
    dataset_test = SnakeSegmentationDataset(transform_image(train=False))

    rand_gen = torch.Generator()
    rand_gen.manual_seed(0)
    indices = torch.randperm(len(dataset), generator=rand_gen).tolist()

    # Use a 80/20 train/test split
    num_examples = len(indices)
    dataset = torch.utils.data.Subset(dataset, indices[: -int(num_examples * 0.2)])
    dataset_test = torch.utils.data.Subset(
        dataset_test, indices[-int(num_examples * 0.2) :]
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # Set up the segmentation model for fine tuning
    model = get_segmentation_model(num_classes)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 20

    eval_results = []
    training_metrics = []

    # start training
    for epoch in range(num_epochs):
        # During training, write results to tensorboard
        metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        WRITER.add_scalar("Segmentation/train loss", metrics.meters["loss"].avg, epoch)
        training_metrics.append(metrics)
        lr_scheduler.step()
        results = evaluate(model, data_loader_test, device=device)
        WRITER.add_scalars("Segmentation/Avg Precision", {"IoU=0.50": results.coco_eval['bbox'].stats[1], "IoU=0.50:0.95": results.coco_eval['bbox'].stats[0]}, epoch)
        WRITER.add_scalars("Segmentation/Avg Recall", {"IoU=0.50:0.95": results.coco_eval['bbox'].stats[8]}, epoch)
        eval_results.append(results)
    return model, results, training_metrics


model, results, metrics, = main()
