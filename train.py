import torch
from torch.utils.data import DataLoader
from model import YOLOScratch
from dataset import YOLODataset
from loss import YOLOLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = YOLODataset("dataset/images/train","dataset/labels/train")
train_dl = DataLoader(train_ds,16,shuffle=True,collate_fn=lambda x:tuple(zip(*x)))

model = YOLOScratch(5).to(DEVICE)
criterion = YOLOLoss()
opt = torch.optim.Adam(model.parameters(),1e-3)

for e in range(100):
    total=0
    for imgs,tgts in train_dl:
        imgs=torch.stack(imgs).to(DEVICE)
        p=model(imgs)
        loss=criterion(p,tgts)
        opt.zero_grad(); loss.backward(); opt.step()
        total+=loss.item()
    print(f"Epoch {e} | Loss {total:.3f}")

torch.save(model.state_dict(),"yolo_scratch.pth")
