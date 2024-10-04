import json
import time
import os
from glob import glob
import random

import torch
import torch.cuda.amp as amp
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn import CrossEntropyLoss, Linear
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms as T

DATA_PATH = os.path.join(os.environ["DATA_PATH"], "aicrowd", "snakes")
LOG_PATH = os.path.join(DATA_PATH, "aicrowd", "snakes", "logs")

WRITER = SummaryWriter(log_dir=LOG_PATH)

SEGMENTATION_MODEL_PATH = os.path.join(
    os.environ["DATA_PATH"], "aicrowd/snakes/segm_model", "snake_seg.pt"
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define transforms for training and validation
image_xforms = {
    "train": T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.Normalize([128, 120, 105], [54, 51, 50]),
        ]
    ),
    "test": T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize([128, 120, 105], [54, 51, 50]),
        ]
    ),
}


def load_segmentation_model(model_path: str):
    seg_model = torch.load(model_path)
    seg_model.to(DEVICE)
    seg_model.eval()
    return seg_model


class SnakeClfData(Dataset):
    def __init__(
        self, file_names, label_to_id, transformation=None, segmentation_model_path=None
    ):
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
        image = T.ToTensor()(image)
        image = image.to(DEVICE)
        image_id = image_file.split("/")[-2].split("-")[1]
        image_id = self.label_to_id[image_id]
        if self.seg_model:
            segmentation = self.seg_model(image.unsqueeze(0))
            # Test if a segmentation was generated
            if segmentation[0]["boxes"].numel() != 0:
                mask = segmentation[0]["boxes"].to(torch.int)[0]
                mask_height = mask[3] - mask[1]
                mask_width = mask[2] - mask[0]
                mask_top = mask[1]
                mask_left = mask[0]
                image = T.functional.crop(
                    image, mask_top, mask_left, mask_height, mask_width
                )
        if self.transformation is not None:
            image = self.transformation(image)
        return image, image_id


# Load the mapping for label to index and human readable name
with open(os.path.join(DATA_PATH, "class_idx_mapping.csv"), "r") as f_in:
    classes = [cl.split(",") for cl in f_in.readlines()[1:]]
    label_to_snake_name = {_cl[1].rstrip(): _cl[0] for _cl in classes}
    label_to_idx = {label: i for i, label in enumerate(label_to_snake_name)}

# Set up dataloader for train/test sets
all_samples = glob(os.path.join(DATA_PATH, "train") + "/*/*", recursive=True)

# Load the jpegs are corrupt
with open(os.path.join(DATA_PATH, "corrupt_img_files.json"), "r") as f_in:
    corrupt_images = json.load(f_in)

# Remove corrupt images from samples
valid_samples = list(set(all_samples).difference(set(corrupt_images)))

# Find the sample indices to create a stratified train/test split
X = []
y = []
for sample in valid_samples:
    split_sample = sample.split("/train/")
    y_sample, X_sample = split_sample[1].split("/")
    X.append(X_sample)
    y.append(y_sample)

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_indices, test_indices = next(strat_split.split(X, y))


# Create the train and test datasets and dataloaders
dataset = {
    x: SnakeClfData(
        valid_samples,
        label_to_idx,
        image_xforms[x],
        # segmentation_model_path=SEGMENTATION_MODEL_PATH,
    )
    for x in ["train", "test"]
}


shuffle_data = {"train": True, "test": False}
phase_indices = {"train": train_indices, "test": test_indices}
# indices = list(range(len(valid_samples)))
# random.seed(0)
# random.shuffle(indices)
# phase_indices = {"train": indices[:5000], "test": indices[5000:6000]}
batch_size = 40

data_loader = {
    x: DataLoader(
        Subset(dataset[x], phase_indices[x]),
        batch_size=batch_size,
        shuffle=shuffle_data[x],
    )
    for x in ["train", "test"]
}


# Load resnet model and tune for specific number of classes
clf_model = models.resnet101(pretrained=True)
in_features = clf_model.fc.in_features
clf_model.fc = Linear(in_features, len(classes))
clf_model = clf_model.to(DEVICE)

crit = CrossEntropyLoss()

optimizer = torch.optim.SGD(clf_model.parameters(), lr=0.005, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

scaler = amp.GradScaler()

num_of_epochs = 10
metrics = []
for epoch in range(num_of_epochs):
    tic = time.time()
    print("-" * 20)
    print(f"Epoch: {epoch + 1}/{num_of_epochs}")

    for phase in ["train", "test"]:
        if phase == "train":
            clf_model.train()
        else:
            clf_model.eval()

        total_loss = 0.0
        correct = 0
        all_preds = []
        all_labels = []

        for inputs, labels in data_loader[phase]:
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Only calculate gradients in the training phase
            with torch.set_grad_enabled(phase == "train"):
                # with amp.autocast():
                outputs = clf_model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = crit(outputs, labels)

                if phase == "train":
                    # scaler.scale(loss).backward()
                    # scaler.step(optimizer)
                    # scaler.update()
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * inputs.shape[0]
            correct += torch.sum(preds == labels)
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())

        if phase == "train":
            lr_scheduler.step()
            toc = time.time()

        epoch_elapsed_time = toc - tic

        epoch_loss = total_loss / len(phase_indices[phase])
        epoch_accuracy = correct.double() / len(phase_indices[phase])
        epoch_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        metrics.append((epoch, [epoch_accuracy, epoch_precision, epoch_recall, epoch_f1]))
        # WRITER.add_scalar(f"Classification {phase} loss:", loss, epoch)
        # WRITER.add_scalar(f"Classification {phase} accuracy:", loss, epoch)

        print(f"Phase: {phase}, elapsed time: {epoch_elapsed_time}")
        print(
                f"Epoch loss: {epoch_loss}, accuracy: {epoch_accuracy}, correct: {correct}, \nprecision: {epoch_precision}, recall: {epoch_recall}, f1: {epoch_f1}"
        )
