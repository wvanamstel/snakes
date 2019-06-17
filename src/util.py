import numpy as np

from collections import defaultdict
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
                    "s": np.var(widths)
                    },
                "height": {
                    "m": np.average(heights),
                    "s": np.var(heights)
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

def _find_average(data, axis):
    return np.average([np.average(im, axis=axis) for im in data])

def _find_var(data, axis):
    return np.average([np.var(im, axis=axis) for im in  data])

# Can also use loaders from torchvision
class SnakeData(Dataset):
    def __init__(self, file_names, transformation=None):
        self.file_names = file_names
        self.transformation = transformation

    def __len__(self):
        return len(self.file_names)

    def __get_item__(self, idx):
        image_file = self.file_names[idx]
        image = Image.open(image_file).convert("RGB")
        label = image_file.spli("/")[-2].split("-")[1]
        if self.transformation:
            self.transformation(image)
        return image, label
