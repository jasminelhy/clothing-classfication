import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataSet(Dataset):
    """Customize the dataset"""

    def __init__(self, images_path: list, images_class: list, transform1=None,transform2=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform1 = transform1
        self.transform2 = transform2


    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert("RGB")
        # RGB indicates a color picture and L indicates a grayscale picture
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform1 is not None:
            img = self.transform1(img)
        if self.transform2 is not None:
            img = np.array(img)
            img_fred = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)# Convert RGB pictures to YCbCr format
            img_fred =self.transform2(img_fred)
        return img_fred, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
