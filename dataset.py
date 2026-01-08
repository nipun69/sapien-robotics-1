import os, torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class YOLODataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.files = sorted(os.listdir(img_dir))
        self.tf = T.Compose([
            T.Resize((416,416)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.tf(Image.open(f"{self.img_dir}/{self.files[i]}").convert("RGB"))
        lbl = self.files[i].replace(".jpg",".txt")

        targets = []
        with open(f"{self.lbl_dir}/{lbl}") as f:
            for l in f:
                targets.append(list(map(float,l.split())))

        return img, torch.tensor(targets)
