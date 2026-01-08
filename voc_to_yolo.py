import os
import xml.etree.ElementTree as ET
from PIL import Image

# ==================================================
# DATASET ROOT (RELATIVE TO intern/)
# ==================================================
VOC_ROOT = "voc_download/VOCdevkit2007/VOC2007"

CLASSES = ["person", "car", "dog", "bicycle", "chair"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}
# ==================================================


def convert(split):
    imgset_path = f"{VOC_ROOT}/ImageSets/Main/{split}.txt"

    img_out = f"dataset/images/{split}"
    lbl_out = f"dataset/labels/{split}"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    with open(imgset_path) as f:
        image_ids = [x.strip() for x in f.readlines()]

    for idx, img_id in enumerate(image_ids):
        img_path = f"{VOC_ROOT}/JPEGImages/{img_id}.jpg"
        ann_path = f"{VOC_ROOT}/Annotations/{img_id}.xml"

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image.save(f"{img_out}/{idx:06d}.jpg")

        tree = ET.parse(ann_path)
        root = tree.getroot()

        yolo_lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in CLASS2ID:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            cls_id = CLASS2ID[cls_name]
            yolo_lines.append(f"{cls_id} {x_center} {y_center} {bw} {bh}")

        with open(f"{lbl_out}/{idx:06d}.txt", "w") as f:
            f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)

    convert("train")
    convert("val")

    with open("dataset/classes.txt", "w") as f:
        f.write("\n".join(CLASSES))
