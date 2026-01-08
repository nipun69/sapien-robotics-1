import torch, torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, p, targets, S=13):
        obj = torch.zeros(p.shape[:-1], device=p.device)
        box = torch.zeros_like(p[...,:4])
        cls = torch.zeros_like(p[...,5:])

        for b,t in enumerate(targets):
            for c,x,y,w,h in t:
                gx,gy = int(x*S), int(y*S)
                obj[b,gy,gx]=1
                box[b,gy,gx]=torch.tensor([x,y,w,h],device=p.device)
                cls[b,gy,gx,int(c)]=1

        return (
            self.mse(p[...,:4]*obj.unsqueeze(-1), box*obj.unsqueeze(-1)) +
            self.bce(p[...,4], obj) +
            self.bce(p[...,5:], cls)
        )
