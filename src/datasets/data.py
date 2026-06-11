"""
Improved FSC-147 Dataset loader.

Key fixes compared with the original version:
1. Do NOT apply ColorJitter to density maps.
2. Actually use img_size and zero_shot in resize_and_pad().
3. Keep density-map sum stable after resize/pad to preserve object counts.
4. Support configurable image / annotation / density-map directories.
5. Pad exemplar boxes to a fixed [num_objects, 4] shape for DataLoader stacking.
6. Make tiling augmentation bbox offsets consistent with num_tiles x num_tiles.
7. Clamp bbox coordinates by img_size instead of hard-coded 1024.
8. Provide an optional minimal COCO fallback when pycocotools is unavailable.

Expected FSC147-style directory:
FSC147/
├── annotations/
│   ├── Train_Test_Val_FSC_147.json
│   ├── annotation_FSC147_384.json
│   ├── instances_train.json        # optional, needed for gt_bboxes if available
│   ├── instances_val.json          # optional
│   └── instances_test.json         # optional
├── images_384_VarV2/
└── gt_density_map_adaptive_512_512_object_VarV2/
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

try:
    from pycocotools.coco import COCO  # type: ignore
except Exception:
    COCO = None


class MinimalCOCO:
    """Small fallback for the few COCO APIs used in this dataset."""

    def __init__(self, annotation_file: str):
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.imgs: Dict[int, Dict] = {
            int(img["id"]): img for img in data.get("images", [])
        }
        self.anns: Dict[int, Dict] = {
            int(ann["id"]): ann for ann in data.get("annotations", [])
        }
        self.img_to_ann_ids: Dict[int, List[int]] = {}

        for ann_id, ann in self.anns.items():
            image_id = int(ann["image_id"])
            self.img_to_ann_ids.setdefault(image_id, []).append(ann_id)

    def getAnnIds(self, imgIds: Optional[Iterable[int]] = None):
        if imgIds is None:
            return list(self.anns.keys())

        result: List[int] = []
        for img_id in imgIds:
            result.extend(self.img_to_ann_ids.get(int(img_id), []))
        return result

    def loadAnns(self, ids: Iterable[int]):
        return [self.anns[int(i)] for i in ids if int(i) in self.anns]


def xywh_to_x1y1x2y2(xywh: Sequence[float]) -> List[float]:
    x, y, w, h = xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def _as_box_tensor(boxes, dtype=torch.float32) -> torch.Tensor:
    if boxes is None:
        return torch.zeros((0, 4), dtype=dtype)

    tensor = torch.as_tensor(boxes, dtype=dtype)

    if tensor.numel() == 0:
        return torch.zeros((0, 4), dtype=dtype)

    return tensor.reshape(-1, 4)


def _sanitize_xyxy_boxes(boxes: torch.Tensor) -> torch.Tensor:
    boxes = _as_box_tensor(boxes, dtype=torch.float32)

    if boxes.numel() == 0:
        return boxes

    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])

    return torch.stack([x1, y1, x2, y2], dim=1)


def _pad_or_truncate_boxes(boxes: torch.Tensor, num_boxes: int) -> torch.Tensor:
    boxes = _sanitize_xyxy_boxes(boxes)

    if boxes.shape[0] >= num_boxes:
        return boxes[:num_boxes]

    pad = torch.zeros(
        (num_boxes - boxes.shape[0], 4),
        dtype=boxes.dtype,
        device=boxes.device,
    )

    return torch.cat([boxes, pad], dim=0)


def _scale_boxes(boxes: torch.Tensor, scale_x: float, scale_y: float) -> torch.Tensor:
    boxes = _as_box_tensor(boxes)

    if boxes.numel() == 0:
        return boxes

    scale = torch.tensor(
        [scale_x, scale_y, scale_x, scale_y],
        dtype=boxes.dtype,
        device=boxes.device,
    )

    return boxes * scale


def _clamp_boxes_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    boxes = _as_box_tensor(boxes)

    if boxes.numel() == 0:
        return boxes

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)

    return _sanitize_xyxy_boxes(boxes)


def _resize_preserve_sum(
    density_map: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a [1, H, W] density map while preserving its total count."""

    if density_map.ndim != 3:
        raise ValueError(
            f"density_map must have shape [1, H, W], got {tuple(density_map.shape)}"
        )

    original_sum = density_map.sum()

    resized = F.interpolate(
        density_map.unsqueeze(0),
        size=size,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )[0]

    new_sum = resized.sum()

    if torch.isfinite(new_sum) and new_sum.abs() > 1e-12:
        resized = resized / new_sum * original_sum

    return resized


def _make_tile(tensor: torch.Tensor, num_tiles: int, transform=None) -> torch.Tensor:
    rows = []

    for _ in range(num_tiles):
        cols = []

        for _ in range(num_tiles):
            tile = transform(tensor) if transform is not None else tensor
            cols.append(tile)

        rows.append(torch.cat(cols, dim=-1))

    return torch.cat(rows, dim=-2)


def _extract_exemplar_boxes(annotation_entry: Dict, num_objects: int) -> torch.Tensor:
    """Parse FSC147 exemplar boxes from annotation_FSC147_384.json."""

    if "box_examples_coordinates" not in annotation_entry:
        raise KeyError("Missing key 'box_examples_coordinates' in FSC147 annotation entry.")

    coords = torch.as_tensor(
        annotation_entry["box_examples_coordinates"],
        dtype=torch.float32,
    )

    if coords.numel() == 0:
        return torch.zeros((num_objects, 4), dtype=torch.float32)

    if coords.ndim == 3 and coords.shape[-1] == 2:
        if coords.shape[1] >= 3:
            boxes = coords[:, [0, 2], :].reshape(-1, 4)
        else:
            raise ValueError(f"Unsupported exemplar coordinate shape: {tuple(coords.shape)}")
    elif coords.ndim == 2 and coords.shape[-1] == 4:
        boxes = coords
    else:
        boxes = coords.reshape(-1, 4)

    return _pad_or_truncate_boxes(boxes, num_objects)


def tiling_augmentation(
    img: torch.Tensor,
    bboxes: torch.Tensor,
    resize: T.Resize,
    jitter,
    tile_size: Tuple[torch.Tensor, torch.Tensor],
    hflip_p: float,
    gt_bboxes: Optional[torch.Tensor] = None,
    density_map: Optional[torch.Tensor] = None,
    density_output_size: int = 512,
):
    """Tile image/density map and update bbox coordinates.

    Important:
    ColorJitter is applied only to the RGB image, not to density_map.
    """

    _, img_h, img_w = img.shape

    x_tile, y_tile = tile_size
    target_h, target_w = resize.size

    num_tiles = max(
        int(torch.ceil(x_tile).item()),
        int(torch.ceil(y_tile).item()),
    )
    num_tiles = max(1, num_tiles)

    tiled_img = _make_tile(img, num_tiles, transform=jitter)
    tiled_h, tiled_w = tiled_img.shape[-2], tiled_img.shape[-1]

    img = resize(tiled_img)

    scale_x = float(target_w) / float(tiled_w)
    scale_y = float(target_h) / float(tiled_h)

    bboxes = _scale_boxes(bboxes, scale_x, scale_y)

    if density_map is not None:
        density_map = _make_tile(density_map, num_tiles, transform=None)
        density_map = _resize_preserve_sum(
            density_map,
            (density_output_size, density_output_size),
        )

    if gt_bboxes is not None:
        gt_bboxes = _as_box_tensor(gt_bboxes)
        tiled_gt_boxes = []

        if gt_bboxes.numel() > 0:
            for row in range(num_tiles):
                for col in range(num_tiles):
                    offset = torch.tensor(
                        [
                            col * img_w,
                            row * img_h,
                            col * img_w,
                            row * img_h,
                        ],
                        dtype=gt_bboxes.dtype,
                        device=gt_bboxes.device,
                    )
                    tiled_gt_boxes.append(gt_bboxes + offset)

            gt_bboxes = torch.cat(tiled_gt_boxes, dim=0)
            gt_bboxes = _scale_boxes(gt_bboxes, scale_x, scale_y)
        else:
            gt_bboxes = torch.zeros((0, 4), dtype=torch.float32)

    if torch.rand(1).item() < hflip_p:
        img = TVF.hflip(img)

        bboxes[:, [0, 2]] = target_w - bboxes[:, [2, 0]]

        if density_map is not None:
            density_map = TVF.hflip(density_map)

        if gt_bboxes is not None and gt_bboxes.numel() > 0:
            gt_bboxes[:, [0, 2]] = target_w - gt_bboxes[:, [2, 0]]

    bboxes = _clamp_boxes_xyxy(bboxes, target_w, target_h)

    if gt_bboxes is not None:
        gt_bboxes = _clamp_boxes_xyxy(gt_bboxes, target_w, target_h)

    return img, bboxes, density_map, gt_bboxes


def resize_and_pad(
    img: torch.Tensor,
    bboxes: torch.Tensor,
    density_map: Optional[torch.Tensor] = None,
    gt_bboxes: Optional[torch.Tensor] = None,
    size: int = 1024,
    zero_shot: bool = False,
    train: bool = False,
    density_output_size: int = 512,
):
    """Resize image by aspect ratio, pad to square, and transform boxes/maps."""

    channels, original_height, original_width = img.shape
    longer_dimension = max(original_height, original_width)

    scaling_factor = float(size) / float(longer_dimension)

    bboxes = _sanitize_xyxy_boxes(bboxes)
    gt_bboxes = _sanitize_xyxy_boxes(gt_bboxes) if gt_bboxes is not None else None

    if not zero_shot and not train and bboxes.numel() > 0:
        scaled_bboxes = bboxes * scaling_factor

        widths = scaled_bboxes[:, 2] - scaled_bboxes[:, 0]
        heights = scaled_bboxes[:, 3] - scaled_bboxes[:, 1]

        valid = (widths > 0) & (heights > 0)

        if valid.any():
            a_dim = ((widths[valid].mean() + heights[valid].mean()) / 2).item()

            if a_dim > 1e-6:
                scaling_factor = min(1.0, 80.0 / a_dim) * scaling_factor

    resized_img = F.interpolate(
        img.unsqueeze(0),
        scale_factor=scaling_factor,
        mode="bilinear",
        align_corners=False,
    )

    resized_height = resized_img.shape[2]
    resized_width = resized_img.shape[3]

    pad_height = max(0, int(size) - resized_height)
    pad_width = max(0, int(size) - resized_width)

    padded_img = F.pad(
        resized_img,
        (0, pad_width, 0, pad_height),
        mode="constant",
        value=0,
    )[0]

    bboxes = _scale_boxes(bboxes, scaling_factor, scaling_factor)
    bboxes = _clamp_boxes_xyxy(bboxes, int(size), int(size))

    if gt_bboxes is not None:
        gt_bboxes = _scale_boxes(gt_bboxes, scaling_factor, scaling_factor)
        gt_bboxes = _clamp_boxes_xyxy(gt_bboxes, int(size), int(size))

    padded_density_map = None

    if density_map is not None:
        density_map = _resize_preserve_sum(
            density_map,
            (original_height, original_width),
        )

        resized_density = F.interpolate(
            density_map.unsqueeze(0),
            scale_factor=scaling_factor,
            mode="bilinear",
            align_corners=False,
        )[0]

        padded_density = F.pad(
            resized_density.unsqueeze(0),
            (0, pad_width, 0, pad_height),
            mode="constant",
            value=0,
        )[0]

        padded_density_map = _resize_preserve_sum(
            padded_density,
            (density_output_size, density_output_size),
        )

    if gt_bboxes is None and density_map is None:
        return padded_img, bboxes, scaling_factor

    if padded_density_map is None:
        padded_density_map = torch.zeros(
            (1, density_output_size, density_output_size),
            dtype=img.dtype,
        )

    if gt_bboxes is None:
        gt_bboxes = torch.zeros((0, 4), dtype=torch.float32)

    return padded_img, bboxes, padded_density_map, gt_bboxes, scaling_factor, (
        pad_width,
        pad_height,
    )


def pad_collate(batch):
    """Collate function for train mode."""

    img, bboxes, density_map, image_ids, gt_bboxes = zip(*batch)

    gt_bboxes_pad = pad_sequence(
        list(gt_bboxes),
        batch_first=True,
        padding_value=0,
    )

    return (
        torch.stack(list(img)),
        torch.stack(list(bboxes)),
        torch.stack(list(density_map)),
        torch.stack(list(image_ids)),
        gt_bboxes_pad,
    )


def pad_collate_test(batch):
    """Collate function for val/test mode."""

    img, bboxes, density_map, image_ids, gt_bboxes, scaling_factor, padwh = zip(*batch)

    gt_bboxes_pad = pad_sequence(
        list(gt_bboxes),
        batch_first=True,
        padding_value=0,
    )

    return (
        torch.stack(list(img)),
        torch.stack(list(bboxes)),
        torch.stack(list(density_map)),
        torch.stack(list(image_ids)),
        gt_bboxes_pad,
        torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in scaling_factor]),
        torch.stack([torch.as_tensor(x, dtype=torch.long) for x in padwh]),
    )


class FSC147DATASET(Dataset):
    def __init__(
        self,
        data_path: str,
        img_size: int,
        split: str = "train",
        num_objects: int = 3,
        tiling_p: float = 0.5,
        zero_shot: bool = False,
        return_ids: bool = False,
        training: bool = False,
        horizontal_flip_p: float = 0.5,
        annotations_dir: str = "annotations",
        image_dir: str = "images_384_VarV2",
        density_map_dir: str = "gt_density_map_adaptive_512_512_object_VarV2",
        split_file: str = "Train_Test_Val_FSC_147.json",
        annotation_file: str = "annotation_FSC147_384.json",
        coco_instances_file: Optional[str] = None,
        density_output_size: int = 512,
        allow_missing_coco: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.split = split
        self.data_path = data_path
        self.img_size = int(img_size)
        self.num_objects = int(num_objects)
        self.tiling_p = float(tiling_p)
        self.zero_shot = bool(zero_shot)
        self.return_ids = bool(return_ids)
        self.training = bool(training)
        self.horizontal_flip_p = float(horizontal_flip_p)
        self.annotations_dir = annotations_dir
        self.image_dir = image_dir
        self.density_map_dir = self._resolve_density_map_dir(density_map_dir)
        self.density_output_size = int(density_output_size)
        self.allow_missing_coco = bool(allow_missing_coco)

        self.resize = T.Resize((self.img_size, self.img_size), antialias=True)
        self.jitter = T.RandomApply(
            [T.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8,
        )
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        split_path = os.path.join(
            self.data_path,
            self.annotations_dir,
            split_file,
        )
        annotation_path = os.path.join(
            self.data_path,
            self.annotations_dir,
            annotation_file,
        )

        with open(split_path, "r", encoding="utf-8") as file:
            splits = json.load(file)

            if split not in splits:
                raise KeyError(f"Split {split!r} not found in {split_path}")

            self.image_names = splits[split]

        with open(annotation_path, "r", encoding="utf-8") as file:
            self.annotations = json.load(file)

        if coco_instances_file is None:
            coco_instances_file = f"instances_{split}.json"

        instances_path = os.path.join(
            self.data_path,
            self.annotations_dir,
            coco_instances_file,
        )

        self.labels = None
        self.img_name_to_ori_id = {}

        if os.path.exists(instances_path):
            if COCO is not None:
                self.labels = COCO(instances_path)
            else:
                self.labels = MinimalCOCO(instances_path)

            self.img_name_to_ori_id = self.map_img_name_to_ori_id()

        elif not self.allow_missing_coco:
            raise FileNotFoundError(
                f"COCO instances file not found: {instances_path}. "
                "Set allow_missing_coco=True to continue without gt_bboxes."
            )

    def _resolve_density_map_dir(self, density_map_dir: str) -> str:
        density_path = os.path.join(self.data_path, density_map_dir)
        if os.path.isdir(density_path):
            return density_map_dir

        candidates = (
            "gt_density_map_adaptive_384_VarV2",
            "gt_density_map_adaptive_512_512_object_VarV2",
        )
        for candidate in candidates:
            candidate_path = os.path.join(self.data_path, candidate)
            if os.path.isdir(candidate_path):
                return candidate

        return density_map_dir

    def __len__(self):
        return len(self.image_names)

    def map_img_name_to_ori_id(self) -> Dict[str, int]:
        if self.labels is None:
            return {}

        map_name_2_id = {}

        for _, value in self.labels.imgs.items():
            img_id = int(value["id"])
            img_name = value["file_name"]
            map_name_2_id[img_name] = img_id

        return map_name_2_id

    def get_gt_bboxes(self, idx: int) -> torch.Tensor:
        """Read all GT bboxes for an image from COCO instances file if available."""

        if self.labels is None:
            return torch.zeros((0, 4), dtype=torch.float32)

        image_name = self.image_names[idx]

        if image_name not in self.img_name_to_ori_id:
            return torch.zeros((0, 4), dtype=torch.float32)

        coco_im_id = self.img_name_to_ori_id[image_name]
        anno_ids = self.labels.getAnnIds(imgIds=[coco_im_id])
        annotations = self.labels.loadAnns(anno_ids)

        bboxes = [
            xywh_to_x1y1x2y2(a["bbox"])
            for a in annotations
            if "bbox" in a
        ]

        return _sanitize_xyxy_boxes(_as_box_tensor(bboxes))

    def _load_image(self, image_name: str) -> torch.Tensor:
        image_path = os.path.join(
            self.data_path,
            self.image_dir,
            image_name,
        )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        return self.to_tensor(image)

    def _load_density_map(self, image_name: str) -> torch.Tensor:
        npy_name = os.path.splitext(image_name)[0] + ".npy"

        density_path = os.path.join(
            self.data_path,
            self.density_map_dir,
            npy_name,
        )

        if not os.path.exists(density_path):
            raise FileNotFoundError(
                f"Density map not found: {density_path}. "
                "Check density_map_dir or file naming."
            )

        density = np.load(density_path).astype(np.float32)
        density_tensor = torch.from_numpy(density)

        if density_tensor.ndim == 2:
            density_tensor = density_tensor.unsqueeze(0)
        elif density_tensor.ndim == 3 and density_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected density map shape [H, W] or [1, H, W], got {density_tensor.shape}"
            )

        return density_tensor

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]

        img = self._load_image(image_name)
        gt_bboxes = self.get_gt_bboxes(idx)

        if image_name not in self.annotations:
            raise KeyError(f"Image {image_name!r} not found in annotation file.")

        bboxes = _extract_exemplar_boxes(
            self.annotations[image_name],
            self.num_objects,
        )

        density_map = self._load_density_map(image_name)

        if self.split == "train":
            tiled = False

            _, original_height, original_width = img.shape
            longer_dimension = max(original_height, original_width)
            rough_scaling_factor = float(self.img_size) / float(longer_dimension)
            bboxes_resized = bboxes * rough_scaling_factor

            box_widths = bboxes_resized[:, 2] - bboxes_resized[:, 0]
            box_heights = bboxes_resized[:, 3] - bboxes_resized[:, 1]

            valid_exemplars = (box_widths > 0) & (box_heights > 0)

            can_tile = False

            if valid_exemplars.any():
                can_tile = bool(
                    box_widths[valid_exemplars].mean() > 30
                    and box_heights[valid_exemplars].mean() > 30
                )

            if can_tile and torch.rand(1).item() < self.tiling_p:
                tiled = True
                tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)

                img, bboxes, density_map, gt_bboxes = tiling_augmentation(
                    img=img,
                    bboxes=bboxes,
                    resize=self.resize,
                    jitter=self.jitter,
                    tile_size=tile_size,
                    hflip_p=self.horizontal_flip_p,
                    gt_bboxes=gt_bboxes,
                    density_map=density_map,
                    density_output_size=self.density_output_size,
                )

            else:
                img = self.jitter(img)

                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                    img=img,
                    bboxes=bboxes,
                    density_map=density_map,
                    gt_bboxes=gt_bboxes,
                    size=self.img_size,
                    zero_shot=self.zero_shot,
                    train=True,
                    density_output_size=self.density_output_size,
                )

            if not tiled and torch.rand(1).item() < self.horizontal_flip_p:
                img = TVF.hflip(img)
                density_map = TVF.hflip(density_map)

                bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]

                if gt_bboxes.numel() > 0:
                    gt_bboxes[:, [0, 2]] = self.img_size - gt_bboxes[:, [2, 0]]

                bboxes = _clamp_boxes_xyxy(
                    bboxes,
                    self.img_size,
                    self.img_size,
                )
                gt_bboxes = _clamp_boxes_xyxy(
                    gt_bboxes,
                    self.img_size,
                    self.img_size,
                )

        else:
            img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                img=img,
                bboxes=bboxes,
                density_map=density_map,
                gt_bboxes=gt_bboxes,
                size=self.img_size,
                zero_shot=self.zero_shot,
                train=False,
                density_output_size=self.density_output_size,
            )

        bboxes = _pad_or_truncate_boxes(
            _clamp_boxes_xyxy(bboxes, self.img_size, self.img_size),
            self.num_objects,
        )
        gt_bboxes = _clamp_boxes_xyxy(
            gt_bboxes,
            self.img_size,
            self.img_size,
        )

        img = self.normalize(img)

        image_id = torch.tensor(idx, dtype=torch.long)

        if self.split == "train" or self.training:
            return img, bboxes, density_map, image_id, gt_bboxes

        return (
            img,
            bboxes,
            density_map,
            image_id,
            gt_bboxes,
            torch.tensor(float(scaling_factor), dtype=torch.float32),
            torch.tensor(padwh, dtype=torch.long),
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test for improved FSC147 dataset loader."
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to FSC147 root directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--density-map-dir",
        default="gt_density_map_adaptive_512_512_object_VarV2",
    )
    parser.add_argument(
        "--image-dir",
        default="images_384_VarV2",
    )
    parser.add_argument(
        "--annotations-dir",
        default="annotations",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--no-missing-coco",
        action="store_true",
        help="Raise error if instances_{split}.json is missing.",
    )

    return parser


def main():
    args = _build_argparser().parse_args()

    dataset = FSC147DATASET(
        data_path=args.data_path,
        img_size=args.img_size,
        split=args.split,
        num_objects=args.num_objects,
        density_map_dir=args.density_map_dir,
        image_dir=args.image_dir,
        annotations_dir=args.annotations_dir,
        allow_missing_coco=not args.no_missing_coco,
        training=(args.split == "train"),
    )

    collate_fn = pad_collate if args.split == "train" else pad_collate_test

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    print(f"Loaded split={args.split!r}, dataset size={len(dataset)}")

    for i, item in enumerate(batch):
        if torch.is_tensor(item):
            print(f"batch[{i}]: shape={tuple(item.shape)}, dtype={item.dtype}")
        else:
            print(f"batch[{i}]: {type(item)}")

    density_map = batch[2]
    print("density sums:", density_map.sum(dim=(1, 2, 3)))


if __name__ == "__main__":
    main()
