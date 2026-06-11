import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms as T

from arg_parser import get_argparser
from src.datasets.data import FSC147DATASET, pad_collate_test, resize_and_pad
from src.models.DGECO import build_model
from src.utils.postprocess import filter_detections
from src.utils.verification import verify_detections


DATASETS = {"fsc147": FSC147DATASET}
DEFAULT_SAM3_CHECKPOINT = (
    "src/models/backbones/sam3/checkpoints/sam3.pt"
)


def parse_boxes(value: str) -> torch.Tensor:
    boxes = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        coords = [float(x.strip()) for x in item.split(",")]
        if len(coords) != 4:
            raise argparse.ArgumentTypeError(
                f"Each box must have 4 comma-separated values, got {item!r}"
            )
        x1, y1, x2, y2 = coords
        boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

    if not boxes:
        raise argparse.ArgumentTypeError("At least one exemplar box is required.")

    return torch.tensor(boxes, dtype=torch.float32)


def build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, type=str)
    pre_args, _ = pre_parser.parse_known_args()

    parent = get_argparser()
    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            parent.set_defaults(**json.load(f))

    parser = argparse.ArgumentParser(
        "DINO-SAM-GECO2 inference",
        parents=[parent],
        conflict_handler="resolve",
    )
    parser.add_argument("--config", default=pre_args.config, type=str)
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to DGECO checkpoint. Defaults to model_path/model_name.pth.",
    )
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--image", default=None, type=str)
    parser.add_argument(
        "--boxes",
        default=None,
        type=parse_boxes,
        help='Single-image exemplar boxes, e.g. "x1,y1,x2,y2;x1,y1,x2,y2".',
    )
    parser.add_argument("--output-dir", default="outputs/inference", type=str)
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-coco-json", action="store_true")
    parser.add_argument(
        "--coco-eval",
        action="store_true",
        help="Run COCO bbox evaluation against annotations/instances_{split}.json.",
    )
    parser.add_argument(
        "--coco-category-id",
        default=1,
        type=int,
        help="Category id assigned to model detections for COCO bbox evaluation.",
    )
    parser.add_argument(
        "--coco-max-dets",
        default=100,
        type=int,
        help="Third maxDets value used by COCOeval, e.g. 100 or 300.",
    )
    parser.add_argument("--sam3-mask", action="store_true")
    parser.add_argument(
        "--sam3-checkpoint",
        default=DEFAULT_SAM3_CHECKPOINT,
        type=str,
    )
    parser.add_argument(
        "--sam3-box-point",
        action="store_true",
        default=True,
        help="Use predicted box plus center point as SAM3 prompts.",
    )
    parser.add_argument(
        "--no-sam3-box-point",
        action="store_false",
        dest="sam3_box_point",
        help="Use only predicted boxes as SAM3 prompts.",
    )
    parser.add_argument("--save-mask-png", action="store_true")
    return parser


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def record_inference_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().astimezone()
    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "args": {key: to_jsonable(value) for key, value in sorted(vars(args).items())},
    }

    latest_path = output_dir / "args.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def get_device(args) -> torch.device:
    if torch.cuda.is_available():
        gpu = int(getattr(args, "gpu", 0))
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def count_valid_gt_boxes(gt_bboxes_i: torch.Tensor) -> int:
    if gt_bboxes_i.numel() == 0:
        return 0
    gt_bboxes_i = gt_bboxes_i.reshape(-1, 4)
    valid_mask = torch.logical_not((gt_bboxes_i == 0).all(dim=1))
    return int(valid_mask.sum().item())


def normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    has_module = all(k.startswith("module.") for k in state_dict.keys())
    if has_module:
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_dgeco(args, device: torch.device) -> torch.nn.Module:
    args.training = True
    model = build_model(args).to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.model_path, f"{args.model_name}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DGECO checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(normalize_checkpoint_keys(state_dict), strict=False)
    model.eval()
    return model


def select_candidate_features(output_i: Dict[str, torch.Tensor], indices: torch.Tensor):
    features = output_i.get("candidate_features")
    if features is None:
        return None
    features = features.reshape(-1, features.shape[-1])
    if indices.numel() == 0:
        return features.new_zeros((0, features.shape[-1]))
    indices = indices.to(device=features.device)
    if int(indices.max().item()) >= features.shape[0]:
        return None
    return features[indices]


def postprocess_outputs(
    output_i: Dict[str, torch.Tensor],
    *,
    score_threshold: float,
    score_ratio: float,
    threshold_mode: str,
    score_quantile: float,
    min_score_gap: float,
    pre_nms_topk: int,
    max_detections: int,
    nms_iou: float,
    min_box_area: float,
    max_box_area: float,
    adaptive_sparse_score_ratio: float = 0.50,
    adaptive_dense_score_ratio: float = 0.45,
    adaptive_sparse_nms_iou: float = 0.25,
    adaptive_dense_nms_iou: float = 0.25,
    adaptive_dense_candidate_threshold: int = 128,
    exemplar_boxes: Optional[torch.Tensor] = None,
    verification_mode: str = "none",
    verification_threshold: float = 0.0,
    verification_topk: int = 0,
    verification_min_area_ratio: float = 0.0,
    verification_max_area_ratio: float = 0.0,
    verification_filter_mode: str = "hard",
    verification_score_gamma: float = 0.0,
    verification_hard_candidate_limit: int = 0,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, ...]:
    boxes, scores, postprocess_stats, keep_indices = filter_detections(
        output_i,
        score_threshold=score_threshold,
        score_ratio=score_ratio,
        threshold_mode=threshold_mode,
        score_quantile=score_quantile,
        min_score_gap=min_score_gap,
        pre_nms_topk=pre_nms_topk,
        max_detections=max_detections,
        nms_iou=nms_iou,
        min_box_area=min_box_area,
        max_box_area=max_box_area,
        adaptive_sparse_score_ratio=adaptive_sparse_score_ratio,
        adaptive_dense_score_ratio=adaptive_dense_score_ratio,
        adaptive_sparse_nms_iou=adaptive_sparse_nms_iou,
        adaptive_dense_nms_iou=adaptive_dense_nms_iou,
        adaptive_dense_candidate_threshold=adaptive_dense_candidate_threshold,
        return_stats=True,
        return_indices=True,
    )
    candidate_features = select_candidate_features(output_i, keep_indices)
    boxes, scores, verification_stats = verify_detections(
        boxes,
        scores,
        exemplar_boxes=exemplar_boxes,
        candidate_features=candidate_features,
        exemplar_features=output_i.get("exemplar_features"),
        mode=verification_mode,
        threshold=verification_threshold,
        topk=verification_topk,
        min_area_ratio=verification_min_area_ratio,
        max_area_ratio=verification_max_area_ratio,
        filter_mode=verification_filter_mode,
        score_gamma=verification_score_gamma,
        hard_candidate_limit=verification_hard_candidate_limit,
        return_stats=True,
    )
    if return_stats:
        return boxes, scores, postprocess_stats, verification_stats
    return boxes, scores


def boxes_to_original_coordinates(
    boxes_norm: torch.Tensor,
    *,
    image_size: int,
    scaling_factor: float,
    padwh: Sequence[int],
    original_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if boxes_norm.numel() == 0:
        return boxes_norm.new_zeros((0, 4)), torch.zeros(
            (0,), dtype=torch.bool, device=boxes_norm.device
        )

    boxes = boxes_norm * float(image_size)
    pad_w, pad_h = int(padwh[0]), int(padwh[1])
    valid_w = float(image_size - pad_w)
    valid_h = float(image_size - pad_h)

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    valid = (centers[:, 0] < valid_w) & (centers[:, 1] < valid_h)
    boxes = boxes[valid]

    if boxes.numel() == 0:
        return boxes_norm.new_zeros((0, 4)), valid

    boxes = boxes / float(scaling_factor)
    if original_size is not None:
        width, height = original_size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    return boxes, valid


def tensor_to_original_image(img_tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = (img_tensor * std + mean).clamp(0, 1)
    array = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def draw_visualization(
    image: Image.Image,
    pred_boxes: Sequence[Sequence[float]],
    exemplar_boxes: Optional[Sequence[Sequence[float]]],
    output_path: Path,
) -> None:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    if exemplar_boxes is not None:
        for box in exemplar_boxes:
            draw.rectangle(list(map(float, box)), outline="red", width=3)
    for box in pred_boxes:
        draw.rectangle(list(map(float, box)), outline="orange", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(output_path)


class SAM3MaskGenerator:
    def __init__(self, checkpoint_path: str, device: torch.device):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

        sam3_root = Path(__file__).resolve().parent / "src" / "models" / "backbones"
        sys.path.insert(0, str(sam3_root))

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = str(device)
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            enable_inst_interactivity=True,
            device=self.device,
        )
        self.processor = Sam3Processor(self.model, device=self.device)

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        boxes_xyxy: Sequence[Sequence[float]],
        *,
        use_center_point: bool,
    ) -> List[Dict[str, object]]:
        state = self.processor.set_image(image)
        results = []

        for box in boxes_xyxy:
            x1, y1, x2, y2 = [float(v) for v in box]
            kwargs = {"box": [x1, y1, x2, y2]}
            if use_center_point:
                kwargs["point_coords"] = [[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]
                kwargs["point_labels"] = [1]

            masks, scores, _ = self.model.predict_inst(state, **kwargs)
            masks_np = self._to_numpy(masks)
            scores_np = self._to_numpy(scores).reshape(-1)
            if masks_np.ndim == 4:
                masks_np = masks_np[0]

            if masks_np.shape[0] == 0:
                continue

            best_idx = int(np.argmax(scores_np))
            mask = masks_np[best_idx].astype(np.float32)
            mask = self._crop_mask_to_box(mask, [x1, y1, x2, y2])
            results.append(
                {
                    "mask": mask,
                    "score": float(scores_np[best_idx]),
                    "box": mask_to_bbox(mask),
                }
            )

        return results

    @staticmethod
    def _to_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _crop_mask_to_box(mask: np.ndarray, box: Sequence[float]) -> np.ndarray:
        h, w = mask.shape[-2:]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        cropped = np.zeros_like(mask)
        if x2 > x1 and y2 > y1:
            cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return cropped


def mask_to_bbox(mask: np.ndarray, threshold: float = 0.5) -> List[int]:
    ys, xs = np.where(mask > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def save_masks(
    masks: Sequence[Dict[str, object]],
    output_dir: Path,
    stem: str,
) -> List[str]:
    paths = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(masks):
        mask = np.asarray(item["mask"])
        path = output_dir / f"{stem}_{idx:04d}.png"
        Image.fromarray((mask > 0.5).astype(np.uint8) * 255).save(path)
        paths.append(str(path))
    return paths


def make_coco_record(
    *,
    image_id: int,
    image_name: str,
    boxes_xyxy: Sequence[Sequence[float]],
    scores: Sequence[float],
    start_anno_id: int,
    category_id: int = 1,
) -> Tuple[Dict[str, object], List[Dict[str, object]], int]:
    image_info = {"id": int(image_id), "file_name": image_name}
    annotations = []
    anno_id = start_anno_id
    for box, score in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        annotations.append(
            {
                "id": anno_id,
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "score": float(score),
            }
        )
        anno_id += 1
    return image_info, annotations, anno_id


def make_coco_detections(
    *,
    image_id: int,
    boxes_xyxy: Sequence[Sequence[float]],
    scores: Sequence[float],
    category_id: int,
) -> List[Dict[str, object]]:
    detections = []
    for box, score in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0.0 or h <= 0.0:
            continue
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [x1, y1, w, h],
                "score": float(score),
            }
        )
    return detections


def get_instances_path(dataset: FSC147DATASET, split: str) -> Path:
    return (
        Path(dataset.data_path)
        / dataset.annotations_dir
        / f"instances_{split}.json"
    )


def run_coco_bbox_eval(
    *,
    instances_path: Path,
    detections: Sequence[Dict[str, object]],
    image_ids: Sequence[int],
    output_dir: Path,
    category_id: int,
    max_dets: int,
) -> Optional[Dict[str, float]]:
    if not instances_path.exists():
        print(f"Skipping COCO bbox eval: instances file not found: {instances_path}")
        return None

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        print(f"Skipping COCO bbox eval: pycocotools is not installed ({exc}).")
        return None

    coco_gt = COCO(str(instances_path))
    if detections:
        coco_dt = coco_gt.loadRes(list(detections))
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": coco_gt.dataset.get("images", []),
            "categories": coco_gt.dataset.get("categories", []),
            "annotations": [],
        }
        coco_dt.createIndex()

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = sorted({int(image_id) for image_id in image_ids})
    coco_eval.params.catIds = [int(category_id)]
    coco_eval.params.maxDets = [1, 10, max(10, int(max_dets))]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_max",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = {name: float(value) for name, value in zip(names, coco_eval.stats)}
    metrics.update(
        {
            "image_count": int(len(set(image_ids))),
            "detection_count": int(len(detections)),
            "category_id": int(category_id),
            "max_dets": int(max(10, int(max_dets))),
        }
    )

    metrics_path = output_dir / "coco_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved COCO bbox metrics to {metrics_path}")
    return metrics


def run_dataset(args, model: torch.nn.Module, device: torch.device) -> None:
    if args.split is None:
        raise ValueError("Dataset mode requires --split train, --split val, or --split test.")

    dataset_cls = DATASETS[args.dataset]
    dataset = dataset_cls(
        args.data_path,
        args.image_size,
        split=args.split,
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
        training=False,
    )
    if args.split == "train":
        # FSC147DATASET keys augmentation/return shape off split == "train".
        # For inference metrics we need the train image list with deterministic eval transforms.
        dataset.split = "test"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate_test,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    coco = {
        "categories": [{"id": int(args.coco_category_id), "name": "fg"}],
        "images": [],
        "annotations": [],
    }
    coco_detections: List[Dict[str, object]] = []
    coco_eval_image_ids: List[int] = []
    anno_id = 1
    sam3 = SAM3MaskGenerator(args.sam3_checkpoint, device) if args.sam3_mask else None

    seen = 0
    ae = 0.0
    se = 0.0
    with pred_path.open("w", encoding="utf-8") as f:
        for batch in loader:
            img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh = batch
            img = img.to(device)
            bboxes = bboxes.to(device)

            with torch.inference_mode():
                outputs, _, _, _, _aux = model(img, bboxes)

            for idx in range(img.shape[0]):
                dataset_idx = int(ids[idx].item())
                image_name = dataset.image_names[dataset_idx]
                image_path = Path(dataset.data_path) / dataset.image_dir / image_name
                pil_image = Image.open(image_path).convert("RGB")

                exemplar_boxes = bboxes[idx] / float(args.image_size)
                boxes_norm, scores, postprocess_stats, verification_stats = postprocess_outputs(
                    outputs[idx],
                    score_threshold=args.score_threshold,
                    score_ratio=args.score_ratio,
                    threshold_mode=args.threshold_mode,
                    score_quantile=args.score_quantile,
                    min_score_gap=args.min_score_gap,
                    pre_nms_topk=args.pre_nms_topk,
                    max_detections=args.max_detections,
                    nms_iou=args.nms_iou,
                    min_box_area=args.min_box_area,
                    max_box_area=args.max_box_area,
                    adaptive_sparse_score_ratio=args.adaptive_sparse_score_ratio,
                    adaptive_dense_score_ratio=args.adaptive_dense_score_ratio,
                    adaptive_sparse_nms_iou=args.adaptive_sparse_nms_iou,
                    adaptive_dense_nms_iou=args.adaptive_dense_nms_iou,
                    adaptive_dense_candidate_threshold=args.adaptive_dense_candidate_threshold,
                    exemplar_boxes=exemplar_boxes,
                    verification_mode=args.verification_mode,
                    verification_threshold=args.verification_threshold,
                    verification_topk=args.verification_topk,
                    verification_min_area_ratio=args.verification_min_area_ratio,
                    verification_max_area_ratio=args.verification_max_area_ratio,
                    verification_filter_mode=args.verification_filter_mode,
                    verification_score_gamma=args.verification_score_gamma,
                    verification_hard_candidate_limit=args.verification_hard_candidate_limit,
                    return_stats=True,
                )
                boxes_orig, keep_original = boxes_to_original_coordinates(
                    boxes_norm.detach().cpu(),
                    image_size=args.image_size,
                    scaling_factor=float(scaling_factor[idx].item()),
                    padwh=padwh[idx].tolist(),
                    original_size=pil_image.size,
                )
                scores_cpu = scores.detach().cpu()[keep_original]
                boxes_list = boxes_orig.tolist()
                scores_list = scores_cpu.tolist()

                mask_paths = []
                mask_scores = []
                mask_boxes = []
                if sam3 is not None and boxes_list:
                    mask_results = sam3.generate(
                        pil_image,
                        boxes_list,
                        use_center_point=args.sam3_box_point,
                    )
                    mask_scores = [item["score"] for item in mask_results]
                    mask_boxes = [item["box"] for item in mask_results]
                    if args.save_mask_png:
                        mask_paths = save_masks(
                            mask_results,
                            output_dir / "masks",
                            Path(image_name).stem,
                        )

                gt_count = count_valid_gt_boxes(gt_bboxes[idx])
                density_sum_debug = float(density_map[idx].sum().item())
                pred_count = len(boxes_list)
                err = abs(float(gt_count) - pred_count)
                ae += err
                se += err * err
                seen += 1

                record = {
                    "image": image_name,
                    "count": pred_count,
                    "gt_count": gt_count,
                    "gt_count_source": "gt_bboxes",
                    "density_sum_debug": density_sum_debug,
                    "boxes": boxes_list,
                    "scores": scores_list,
                    "postprocess": postprocess_stats.to_dict(),
                    "verification": verification_stats.to_dict(),
                    "mask_boxes": mask_boxes,
                    "mask_scores": mask_scores,
                    "mask_paths": mask_paths,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if args.save_vis:
                    draw_visualization(
                        pil_image,
                        mask_boxes if mask_boxes else boxes_list,
                        None,
                        output_dir / "vis" / image_name,
                    )

                image_id = dataset.img_name_to_ori_id.get(image_name, dataset_idx)
                if args.coco_eval:
                    coco_eval_image_ids.append(int(image_id))
                    coco_detections.extend(
                        make_coco_detections(
                            image_id=image_id,
                            boxes_xyxy=boxes_list,
                            scores=scores_list,
                            category_id=args.coco_category_id,
                        )
                    )

                if args.save_coco_json:
                    image_info, annos, anno_id = make_coco_record(
                        image_id=image_id,
                        image_name=image_name,
                        boxes_xyxy=boxes_list,
                        scores=scores_list,
                        start_anno_id=anno_id,
                        category_id=args.coco_category_id,
                    )
                    coco["images"].append(image_info)
                    coco["annotations"].extend(annos)

                if args.max_images is not None and seen >= args.max_images:
                    break

            if args.max_images is not None and seen >= args.max_images:
                break

    if args.save_coco_json:
        with (output_dir / f"dgeco_{args.split}.json").open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)

    if args.coco_eval:
        run_coco_bbox_eval(
            instances_path=get_instances_path(dataset, args.split),
            detections=coco_detections,
            image_ids=coco_eval_image_ids,
            output_dir=output_dir,
            category_id=args.coco_category_id,
            max_dets=args.coco_max_dets,
        )

    denominator = max(seen, 1)
    print(f"Saved predictions to {pred_path}")
    print(f"{args.split} MAE: {ae / denominator:.2f} RMSE: {(se / denominator) ** 0.5:.2f}")


def run_single_image(args, model: torch.nn.Module, device: torch.device) -> None:
    if args.image is None or args.boxes is None:
        raise ValueError("Single-image mode requires --image and --boxes.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    image_tensor = T.ToTensor()(image)
    bboxes = args.boxes
    padded_img, scaled_boxes, scaling_factor = resize_and_pad(
        image_tensor,
        bboxes,
        size=args.image_size,
        zero_shot=args.zero_shot,
        train=False,
    )
    padded_img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        padded_img
    )

    img_batch = padded_img.unsqueeze(0).to(device)
    box_batch = scaled_boxes.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs, _, _, _, _aux = model(img_batch, box_batch)

    exemplar_boxes = box_batch[0] / float(args.image_size)
    boxes_norm, scores, postprocess_stats, verification_stats = postprocess_outputs(
        outputs[0],
        score_threshold=args.score_threshold,
        score_ratio=args.score_ratio,
        threshold_mode=args.threshold_mode,
        score_quantile=args.score_quantile,
        min_score_gap=args.min_score_gap,
        pre_nms_topk=args.pre_nms_topk,
        max_detections=args.max_detections,
        nms_iou=args.nms_iou,
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area,
        adaptive_sparse_score_ratio=args.adaptive_sparse_score_ratio,
        adaptive_dense_score_ratio=args.adaptive_dense_score_ratio,
        adaptive_sparse_nms_iou=args.adaptive_sparse_nms_iou,
        adaptive_dense_nms_iou=args.adaptive_dense_nms_iou,
        adaptive_dense_candidate_threshold=args.adaptive_dense_candidate_threshold,
        exemplar_boxes=exemplar_boxes,
        verification_mode=args.verification_mode,
        verification_threshold=args.verification_threshold,
        verification_topk=args.verification_topk,
        verification_min_area_ratio=args.verification_min_area_ratio,
        verification_max_area_ratio=args.verification_max_area_ratio,
        verification_filter_mode=args.verification_filter_mode,
        verification_score_gamma=args.verification_score_gamma,
        verification_hard_candidate_limit=args.verification_hard_candidate_limit,
        return_stats=True,
    )
    boxes_orig, keep_original = boxes_to_original_coordinates(
        boxes_norm.detach().cpu(),
        image_size=args.image_size,
        scaling_factor=float(scaling_factor),
        padwh=(args.image_size - int(round(image.width * scaling_factor)), args.image_size - int(round(image.height * scaling_factor))),
        original_size=image.size,
    )
    boxes_list = boxes_orig.tolist()
    scores_list = scores.detach().cpu()[keep_original].tolist()

    mask_paths = []
    mask_scores = []
    mask_boxes = []
    if args.sam3_mask and boxes_list:
        sam3 = SAM3MaskGenerator(args.sam3_checkpoint, device)
        mask_results = sam3.generate(
            image,
            boxes_list,
            use_center_point=args.sam3_box_point,
        )
        mask_scores = [item["score"] for item in mask_results]
        mask_boxes = [item["box"] for item in mask_results]
        if args.save_mask_png:
            mask_paths = save_masks(mask_results, output_dir / "masks", Path(args.image).stem)

    record = {
        "image": args.image,
        "count": len(boxes_list),
        "boxes": boxes_list,
        "scores": scores_list,
        "postprocess": postprocess_stats.to_dict(),
        "verification": verification_stats.to_dict(),
        "mask_boxes": mask_boxes,
        "mask_scores": mask_scores,
        "mask_paths": mask_paths,
    }
    with (output_dir / "prediction.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    if args.save_vis:
        draw_visualization(
            image,
            mask_boxes if mask_boxes else boxes_list,
            bboxes.tolist(),
            output_dir / "vis" / Path(args.image).name,
        )

    print(json.dumps(record, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.split is None and args.image is None:
        parser.error("Use --split for dataset inference or --image with --boxes for single-image inference.")

    record_inference_run(args)

    device = get_device(args)
    model = load_dgeco(args, device)

    if args.image is not None:
        run_single_image(args, model, device)
    else:
        run_dataset(args, model, device)


if __name__ == "__main__":
    main()
