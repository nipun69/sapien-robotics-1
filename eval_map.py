import torch
import numpy as np
from torch.utils.data import DataLoader
from model import YOLOScratch
from dataset import YOLODataset

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 416
GRID_SIZE = 13
IOU_THRESH = 0.5
CONF_THRESH = 0.5
NUM_CLASSES = 5
# =========================================


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


def decode_predictions(pred):
    pred = pred.cpu().numpy()
    boxes = []

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            obj = sigmoid(pred[gy, gx, 4])
            if obj < CONF_THRESH:
                continue

            cls_scores = sigmoid(pred[gy, gx, 5:])
            cls_id = int(np.argmax(cls_scores))
            score = obj * cls_scores[cls_id]

            bx, by, bw, bh = pred[gy, gx, :4]
            cx = (gx + bx) / GRID_SIZE
            cy = (gy + by) / GRID_SIZE

            x1 = (cx - bw / 2) * IMG_SIZE
            y1 = (cy - bh / 2) * IMG_SIZE
            x2 = (cx + bw / 2) * IMG_SIZE
            y2 = (cy + bh / 2) * IMG_SIZE

            boxes.append({
                "cls": cls_id,
                "score": score,
                "bbox": [x1, y1, x2, y2]
            })

    return boxes


def evaluate_map():
    val_ds = YOLODataset(
        "dataset/images/val",
        "dataset/labels/val"
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = YOLOScratch(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("yolo_scratch.pth", map_location=DEVICE))
    model.eval()

    detections = {c: [] for c in range(NUM_CLASSES)}
    ground_truths = {c: [] for c in range(NUM_CLASSES)}

    with torch.no_grad():
        for img, targets in val_loader:
            img = img.to(DEVICE)

            # 🔑 FIX: unpack batch
            targets = targets[0]

            pred = model(img)[0]
            preds = decode_predictions(pred)

            # Ground truth boxes
            for t in targets:
                cls, x, y, w, h = t.tolist()
                x1 = (x - w / 2) * IMG_SIZE
                y1 = (y - h / 2) * IMG_SIZE
                x2 = (x + w / 2) * IMG_SIZE
                y2 = (y + h / 2) * IMG_SIZE
                ground_truths[int(cls)].append([x1, y1, x2, y2])

            # Predictions
            for p in preds:
                detections[p["cls"]].append((p["bbox"], p["score"]))

    APs = []

    for cls in range(NUM_CLASSES):
        preds = sorted(detections[cls], key=lambda x: x[1], reverse=True)
        gts = ground_truths[cls]

        if len(gts) == 0:
            continue

        TP = np.zeros(len(preds))
        FP = np.zeros(len(preds))
        matched = []

        for i, (box, _) in enumerate(preds):
            ious = [compute_iou(box, gt) for gt in gts]
            max_iou = max(ious) if len(ious) else 0
            gt_idx = int(np.argmax(ious)) if len(ious) else -1

            if max_iou >= IOU_THRESH and gt_idx not in matched:
                TP[i] = 1
                matched.append(gt_idx)
            else:
                FP[i] = 1

        TP = np.cumsum(TP)
        FP = np.cumsum(FP)

        recall = TP / (len(gts) + 1e-6)
        precision = TP / (TP + FP + 1e-6)

        AP = np.trapz(precision, recall)
        APs.append(AP)

        print(f"Class {cls} AP: {AP:.4f}")

    mAP = np.mean(APs)
    print("\n==============================")
    print(f"mAP@0.5: {mAP:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    evaluate_map()
