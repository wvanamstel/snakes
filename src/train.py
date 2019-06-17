import numpy as np
import os
import torch

from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util import find_image_stats, SnakeData

DATA_PATH = os.path.join(os.environ["LOCAL_DATA_PATH"], "crowdai", "snakes")
# WRITER = SummaryWriter()
np.random.seed(0)

with open(os.path.join(DATA_PATH, "class_idx_mapping.csv"), "r") as f_in:
    classes = [cl.split(',') for cl in f_in.readlines()[1:]]
    label_to_classname = {_cl[1].rstrip(): _cl[0] for _cl in classes}
train_samples = glob(os.path.join(DATA_PATH, "train") + "/*/*", recursive=True)
np.random.shuffle(train_samples)
stats = find_image_stats(train_samples[:100])



