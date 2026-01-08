import torch,time
from model import YOLOScratch

m=YOLOScratch(5)
m.load_state_dict(torch.load("yolo_scratch.pth"))
m.eval()

x=torch.randn(1,3,416,416)
for _ in range(20): m(x)

t=time.time()
for _ in range(200): m(x)
print("FPS:",200/(time.time()-t))
