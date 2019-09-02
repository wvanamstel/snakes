#{'sizes': {'width': {'m': 730.202, 's': 516.4406308531504}, 'height': {'m': 668.302, 's': 491.36394535618905}}, 'RGB': {'R': {'m': 127.66413400677878, 's': 53.97817510693633}, 'G': {'m': 119.76209593084496, 's': 51.48724926451421}, 'B': {'m': 105.29427634036799, 's': 49.91743975710748}}}
import torch
from torchvision import transforms

USE_CUDA = torch.cuda.is_available()

# Learning parameters
BATCH_SIZE = 128
LR = 3e-4
NUM_EPOCHS = 1

# TRANSFORMATIONS
IMAGE_WIDTH = int(730/3)
IMAGE_HEIGHT = int(668/3)
RGB_MEAN = [0.127, 0.120, 0.105]
RGB_STD = [0.054, 0.052, 0.050]

IMAGE_NORM_TRANS = transforms.Normalize(RGB_MEAN, RGB_STD)

TRANSFORMS = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            IMAGE_NORM_TRANS,
            # transforms.Lambda(lambda x: x.half()),
            ])
            ,
        "val": transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            IMAGE_NORM_TRANS
            ])
        }


