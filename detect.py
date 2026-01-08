import torch,cv2,numpy as np
from model import YOLOScratch

CLASSES=open("dataset/classes.txt").read().splitlines()
S,IMG=13,416

def sig(x): return 1/(1+np.exp(-x))

m=YOLOScratch(len(CLASSES))
m.load_state_dict(torch.load("yolo_scratch.pth"))
m.eval()

cap=cv2.VideoCapture(0)
while True:
    r,f=cap.read()
    if not r: break
    img=cv2.resize(f,(IMG,IMG))/255
    x=torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad(): p=m(x)[0].numpy()
    for y in range(S):
        for xg in range(S):
            if sig(p[y,xg,4])>0.5:
                c=np.argmax(sig(p[y,xg,5:]))
                bx,by,bw,bh=p[y,xg,:4]
                cx,cy=(xg+bx)/S,(y+by)/S
                x1,y1=int((cx-bw/2)*IMG),int((cy-bh/2)*IMG)
                x2,y2=int((cx+bw/2)*IMG),int((cy+bh/2)*IMG)
                cv2.rectangle(f,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(f,CLASSES[c],(x1,y1-5),0,0.6,(0,255,0),2)
    cv2.imshow("YOLO Scratch",f)
    if cv2.waitKey(1)==27: break
