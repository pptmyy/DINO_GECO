import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms as T


def _split_dir(split: str) -> str:
    return split if split.endswith("2017") else f"{split}2017"


def _instances_name(split: str) -> str:
    return f"instances_{_split_dir(split)}.json"


def _xywh_to_xyxy(bbox: Iterable[float]) -> List[float]:
    x, y, w, h = [float(value) for value in bbox]
    return [x, y, x + w, y + h]


def _sanitize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = boxes.reshape(-1, 4).float()
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=1)


def _clamp_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    boxes = _sanitize_boxes(boxes)
    if boxes.numel() == 0:
        return boxes
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    return _sanitize_boxes(boxes)


def _valid_box_mask(boxes: torch.Tensor, min_box_size: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes.device)
    wh = boxes[:, 2:] - boxes[:, :2]
    return (wh[:, 0] >= min_box_size) & (wh[:, 1] >= min_box_size)


def resize_pad_image_and_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    *,
    size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, original_height, original_width = image.shape
    scale = float(size) / float(max(original_height, original_width))
    resized = F.interpolate(
        image.unsqueeze(0),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
    )[0]
    resized_height, resized_width = resized.shape[-2:]
    pad_height = max(0, int(size) - resized_height)
    pad_width = max(0, int(size) - resized_width)
    image = F.pad(resized.unsqueeze(0), (0, pad_width, 0, pad_height), value=0.0)[0]
    boxes = _clamp_boxes(boxes * scale, int(size), int(size))
    return image, boxes


class CocoClassAgnosticDataset(Dataset):
    """COCO instances dataset where every non-crowd object is foreground."""

    def __init__(
        self,
        coco_path: str | Path,
        *,
        split: str = "train2017",
        image_size: int = 1024,
        annotation_file: str | Path | None = None,
        image_dir: str | Path | None = None,
        training: bool = True,
        horizontal_flip_p: float = 0.5,
        color_jitter_p: float = 0.2,
        min_box_size: float = 2.0,
        max_boxes_per_image: int = 0,
        max_images: int | None = None,
        include_crowd: bool = False,
    ) -> None:
        self.coco_path = Path(coco_path)
        self.split = _split_dir(split)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.horizontal_flip_p = float(horizontal_flip_p)
        self.min_box_size = float(min_box_size)
        self.max_boxes_per_image = int(max_boxes_per_image)
        self.include_crowd = bool(include_crowd)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.color_jitter = T.RandomApply(
            [T.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=float(color_jitter_p),
        )

        if annotation_file is None:
            annotation_path = self.coco_path / "annotations" / _instances_name(self.split)
        else:
            annotation_path = Path(annotation_file)
        if annotation_file is not None and not annotation_path.is_absolute():
            annotation_path = self.coco_path / annotation_path

        if image_dir is None:
            image_root = self.coco_path / self.split
        else:
            image_root = Path(image_dir)
        if image_dir is not None and not image_root.is_absolute():
            image_root = self.coco_path / image_root
        self.image_root = image_root

        with annotation_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.images: Dict[int, Dict] = {
            int(image["id"]): image for image in data.get("images", [])
        }
        self.annotations_by_image: Dict[int, List[Dict]] = {
            image_id: [] for image_id in self.images
        }
        for annotation in data.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id in self.annotations_by_image:
                self.annotations_by_image[image_id].append(annotation)

        image_ids = [
            image_id
            for image_id in self.images
            if self._has_valid_annotation(self.annotations_by_image.get(image_id, []))
        ]
        image_ids.sort()
        if max_images is not None:
            image_ids = image_ids[: int(max_images)]
        self.image_ids = image_ids

        if not self.image_ids:
            raise ValueError(f"No valid COCO images found in {annotation_path}")

    def _has_valid_annotation(self, annotations: List[Dict]) -> bool:
        for annotation in annotations:
            if not self.include_crowd and int(annotation.get("iscrowd", 0)) != 0:
                continue
            _x, _y, width, height = [float(value) for value in annotation["bbox"]]
            if width >= self.min_box_size and height >= self.min_box_size:
                return True
        return False

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_boxes(self, image_id: int, width: int, height: int) -> torch.Tensor:
        boxes = []
        for annotation in self.annotations_by_image.get(image_id, []):
            if not self.include_crowd and int(annotation.get("iscrowd", 0)) != 0:
                continue
            box = torch.tensor(_xywh_to_xyxy(annotation["bbox"]), dtype=torch.float32)
            box = _clamp_boxes(box.reshape(1, 4), width, height)[0]
            wh = box[2:] - box[:2]
            if float(wh[0].item()) < self.min_box_size:
                continue
            if float(wh[1].item()) < self.min_box_size:
                continue
            boxes.append(box)

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)

        boxes_tensor = torch.stack(boxes, dim=0)
        if self.max_boxes_per_image > 0 and boxes_tensor.shape[0] > self.max_boxes_per_image:
            boxes_tensor = boxes_tensor[: self.max_boxes_per_image]
        return boxes_tensor

    def __getitem__(self, idx: int):
        image_id = int(self.image_ids[idx])
        image_info = self.images[image_id]
        image_path = self.image_root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes = self._load_boxes(image_id, width, height)
        image_tensor = self.to_tensor(image)

        if self.training and self.horizontal_flip_p > 0:
            if random.random() < self.horizontal_flip_p:
                image_tensor = torch.flip(image_tensor, dims=[2])
                if boxes.numel() > 0:
                    old_x1 = boxes[:, 0].clone()
                    old_x2 = boxes[:, 2].clone()
                    boxes[:, 0] = float(width) - old_x2
                    boxes[:, 2] = float(width) - old_x1
                    boxes = _sanitize_boxes(boxes)

        if self.training:
            image_tensor = self.color_jitter(image_tensor)

        image_tensor, boxes = resize_pad_image_and_boxes(
            image_tensor,
            boxes,
            size=self.image_size,
        )
        valid = _valid_box_mask(boxes, self.min_box_size)
        boxes = boxes[valid]
        image_tensor = self.normalize(image_tensor)
        return image_tensor, boxes, torch.as_tensor(image_id, dtype=torch.long)


def coco_class_agnostic_collate(batch):
    images, boxes, image_ids = zip(*batch)
    return (
        torch.stack(list(images)),
        pad_sequence(list(boxes), batch_first=True, padding_value=0.0),
        torch.stack(list(image_ids)),
    )
