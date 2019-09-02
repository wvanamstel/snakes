import numpy as np

from PIL import Image
from torch.utils.data import Dataset

def find_image_stats(file_name_sample):
    images = [Image.open(im) for im in file_name_sample]
    images_ar = [np.array(im) for im in images]
    rgb = []
    heights = []
    widths = []
    for im in images_ar:
        h, w, d = im.shape
        im.shape = (h*w, d)
        rgb.append(im)
        heights.append(h)
        widths.append(w)

    averages = np.average([np.average(im, axis=0) for im in rgb], axis=0)
    var = np.average([np.std(im, axis=0) for im in rgb], axis=0)
    stats = {
            "sizes":
            {
                "width": {
                    "m": np.average(widths),
                    "s": np.std(widths)
                    },
                "height": {
                    "m": np.average(heights),
                    "s": np.std(heights)
                    }
                },
            "RGB":
            {
                "R": {
                    "m": averages[0],
                    "s": var[0]
                },
                "G": {
                    "m": averages[1],
                    "s": var[1]
                },
                "B": {
                    "m": averages[2],
                    "s": var[2]
                }
            }
        }

    return stats

def find_corrupt_images(file_names):
    invalid = []
    for f_in in file_names:
        try:
            i=Image.open(f_in)
        except OSError:
            invalid.append(f_in)
    return invalid

# Can also use loaders from torchvision
class SnakeData(Dataset):
    def __init__(self, file_names, label_to_idx, transformation=None):
        self.file_names = file_names
        self.transformation = transformation
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_file = self.file_names[idx]
        image = Image.open(image_file).convert("RGB")
        label = self.label_to_idx[image_file.split("/")[-2].split("-")[1]]
        if self.transformation:
            image = self.transformation(image)
        return image, label
