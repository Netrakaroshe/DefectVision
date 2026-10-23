"""
PyTorch Dataset Loader for CR7-DET 7-Class Object Detection.
Offline Dent Pilot Experiment.

Class Mapping:
  0: Background (Torchvision internal)
  1: Inclusion            (CR7-DET YOLO class 0)
  2: Dents                (CR7-DET YOLO class 1)
  3: Oil Spots            (CR7-DET YOLO class 2)
  4: Pits                 (CR7-DET YOLO class 3)
  5: Punching             (CR7-DET YOLO class 4)
  6: Linear (Scratches)   (CR7-DET YOLO class 5)
  7: Macular (Discolor.)  (CR7-DET YOLO class 6)
"""

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

CLASS_NAMES_WITH_BG = [
    "__background__",
    "Inclusion",
    "Dents",
    "Oil Spots",
    "Pits",
    "Punching",
    "Linear",
    "Macular"
]


class CR7DetectionDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str = "train", transforms=None):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transforms = transforms

        records = [json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.samples = [r for r in records if r["split"] == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item["image_path"]
        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size

        boxes = []
        labels = []

        for ann in item["annotations"]:
            cls_id = ann["class_id"]
            cx, cy, bw, bh = ann["bbox_yolo"]

            # Convert YOLO normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)
            x1 = max(0.0, (cx - bw / 2.0) * w_img)
            y1 = max(0.0, (cy - bh / 2.0) * h_img)
            x2 = min(float(w_img), (cx + bw / 2.0) * w_img)
            y2 = min(float(h_img), (cy + bh / 2.0) * h_img)

            # Ensure valid box dimensions
            if x2 > x1 + 1.0 and y2 > y1 + 1.0:
                boxes.append([x1, y1, x2, y2])
                # Shift class ID by +1 for torchvision background=0
                labels.append(cls_id + 1)

        # To tensor
        img_tensor = TF.to_tensor(img)

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
            "stem": item["image_stem"]
        }

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
