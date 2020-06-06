import json
import os
from glob import glob
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TRANSFORMS, BATCH_SIZE, LR, NUM_EPOCHS, USE_CUDA
from model import SimpleCNN
from util import find_image_stats, SnakeData, find_corrupt_images

DATA_PATH = os.path.join(os.environ["LOCAL_DATA_PATH"], "crowdai", "snakes")
# WRITER = SummaryWriter()
np.random.seed(0)

# Get image file names, and remove corrupt files
with open(os.path.join(DATA_PATH, "class_idx_mapping.csv"), "r") as f_in:
    classes = [cl.split(',') for cl in f_in.readlines()[1:]]
    label_to_classname = {_cl[1].rstrip(): _cl[0] for _cl in classes}
    label_to_idx = {i:label for label, i in enumerate(label_to_classname.keys())}
train_samples = set(glob(os.path.join(DATA_PATH, "train") + "/*/*", recursive=True))

with open(os.path.join(DATA_PATH, "corrupt_images.json"), "r") as f_in:
    corrupt_images = set(json.load(f_in))
train_samples = list(train_samples.difference(corrupt_images))
np.random.shuffle(train_samples)

# Create data loaders
num_train_samples = int(0.85*len(train_samples))
num_test_samples = int(0.05*num_train_samples)
val_samples = train_samples[num_train_samples + num_test_samples:]
test_samples = train_samples[num_train_samples: num_train_samples + num_test_samples]
train_samples = train_samples[:num_train_samples]

train_data = SnakeData(train_samples, label_to_idx, TRANSFORMS["train"])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_data = SnakeData(val_samples, label_to_idx, TRANSFORMS["val"])
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Init model
model = SimpleCNN()
# model.half()
if USE_CUDA:
    model.cuda()

optimiser = torch.optim.Adam(model.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

for ep in range(NUM_EPOCHS):
    model.train()
    for idx, (data, labels) in enumerate(train_loader):
        if USE_CUDA:
            data = data.cuda()
            labels = labels.cuda()
        data = Variable(data)
        labels = Variable(labels)
        optimiser.zero_grad()
        output = model(data)
        loss = loss_function(output, labels)
        loss.backward()
        optimiser.step()

        preds = output.data.max(1)[1]
        correct = preds.eq(labels.data).cpu().sum().item()
        accuracy = correct/len(data)
        if not idx % 5*BATCH_SIZE:
            print(f'Train Epoch: {ep+1}/{NUM_EPOCHS} [{idx*len(data)}/{len(train_loader.dataset)} ({100*idx/len(train_loader):.2f}%)]\tLoss: {loss.item():.4f}\tAccuracy: {correct}/{len(data)}, {accuracy:.2f}')
