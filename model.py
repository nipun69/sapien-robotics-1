import torch.nn as nn

def conv(c1, c2):
    return nn.Sequential(
        nn.Conv2d(c1, c2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(c2),
        nn.LeakyReLU(0.1)
    )

class YOLOScratch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            conv(3,16), nn.MaxPool2d(2),
            conv(16,32), nn.MaxPool2d(2),
            conv(32,64), nn.MaxPool2d(2),
            conv(64,128), nn.MaxPool2d(2),
            conv(128,256), nn.MaxPool2d(2),
            conv(256,512)
        )
        self.head = nn.Conv2d(512, 5 + num_classes, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).permute(0,2,3,1)
