import os

import torch
from PIL import Image
from torchvision import transforms as T

DATA_PATH = os.environ["DATA_PATH"]
snake_path = os.path.join(DATA_PATH, "aicrowd/snakes/train")
model_path = os.path.join("/home/wserver/code/crowdai/snakes/segmentation/")

image_paths = [
    "class-4/6d714caaa7a592a2b00db9e4ece69cf5.jpg",
    "class-561/add253d6bff8f9c650e0a26f901eb551.jpg",
    "class-78/78f0f3e3ee7fe869aa5e8e664f1913fe.jpg",
    "class-362/24bb49e48773fddeb726cfe8a83c4b0d.jpg",
    "class-771/2a124485b0a6062fc841e41958852866.jpg",
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(os.path.join(model_path, "snake_seg.pt"))
model.to(device)
model.eval()

results = []
for image in image_paths:
    img = Image.open(os.path.join(snake_path, image)).convert("RGB")
    img = T.ToTensor()(img).unsqueeze(0)
    img = img.to(device)

    output = model(img)

    results.append(output)
