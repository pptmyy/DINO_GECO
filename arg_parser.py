import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _non_negative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _probability(value):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return value


def get_argparser():

    parser = argparse.ArgumentParser("DGECO")

    parser.add_argument('--model_name',default='DGECO2FSCD',type=str)
    parser.add_argument('--model_name_resumed',default='DGECO2FSCD',type=str)
    parser.add_argument('--data_path',default=str(PROJECT_ROOT / 'data' / 'FSC147_384_V2'),type=str)
    parser.add_argument("--dataset", default="fsc147", choices=["fsc147"])
    parser.add_argument("--image_size", default=1024, type=_positive_int)
    parser.add_argument("--batch_size", default=2, type=_positive_int)
    parser.add_argument("--eval_batch_size", default=2, type=_positive_int)
    parser.add_argument("--val_batch_size", default=2, type=_positive_int)
    parser.add_argument("--test_batch_size", default=2, type=_positive_int)
    parser.add_argument("--num_workers", default=8, type=_non_negative_int)
    parser.add_argument("--max_train_batches", default=None, type=_positive_int)
    parser.add_argument("--max_val_batches", default=None, type=_positive_int)
    parser.add_argument("--max_test_batches", default=None, type=_positive_int)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--log_interval", default=50, type=_non_negative_int)
    parser.add_argument("--model_path", default=str(PROJECT_ROOT / "checkpoints"), type=str)
    parser.add_argument("--gpu", default=0, type=_non_negative_int)
    parser.add_argument("--max_grad_norm", default=0.1, type=_non_negative_float)
    parser.add_argument("--score-threshold", default=0.20, type=_probability)
    parser.add_argument("--score-ratio", default=0.50, type=_non_negative_float)
    parser.add_argument(
        "--threshold-mode",
        default="static_ratio",
        choices=["static_ratio", "quantile", "regime_adaptive"],
    )
    parser.add_argument("--score-quantile", default=0.98, type=_probability)
    parser.add_argument("--min-score-gap", default=0.0, type=_non_negative_float)
    parser.add_argument("--pre-nms-topk", default=4096, type=_positive_int)
    parser.add_argument("--max-detections", default=4096, type=_positive_int)
    parser.add_argument("--nms-iou", default=0.30, type=_probability)
    parser.add_argument("--min-box-area", default=0.0, type=_non_negative_float)
    parser.add_argument("--max-box-area", default=0.0, type=_non_negative_float)
    parser.add_argument("--adaptive-sparse-score-ratio", default=0.50, type=_non_negative_float)
    parser.add_argument("--adaptive-dense-score-ratio", default=0.45, type=_non_negative_float)
    parser.add_argument("--adaptive-sparse-nms-iou", default=0.25, type=_probability)
    parser.add_argument("--adaptive-dense-nms-iou", default=0.25, type=_probability)
    parser.add_argument("--adaptive-dense-candidate-threshold", default=128, type=_positive_int)
    parser.add_argument("--verification-mode", default="none", choices=["none", "exemplar_geometry", "feature_similarity"])
    parser.add_argument("--verification-threshold", default=0.0, type=_non_negative_float)
    parser.add_argument("--verification-topk", default=0, type=_non_negative_int)
    parser.add_argument("--verification-min-area-ratio", default=0.0, type=_non_negative_float)
    parser.add_argument("--verification-max-area-ratio", default=0.0, type=_non_negative_float)
    parser.add_argument("--verification-filter-mode", default="hard", choices=["hard", "soft", "sparse_hard"])
    parser.add_argument("--verification-score-gamma", default=0.0, type=_non_negative_float)
    parser.add_argument("--verification-hard-candidate-limit", default=0, type=_non_negative_int)
    parser.add_argument("--dinov3_model_size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dinov3_pretrained_weights",default=str(PROJECT_ROOT / "src" / "models" / "backbones" / "checkpoint" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),type=str)
    parser.add_argument('--backbone',default='DINOV3',type=str)
    parser.add_argument('--reduction',default=16,type=_positive_int)
    parser.add_argument('--kernel_dim',default=1,type=_positive_int)
    parser.add_argument('--num_objects', default=3, type=_positive_int)
    parser.add_argument('--epochs', default=200, type=_positive_int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--lr', default=5e-5, type=_non_negative_float)
    parser.add_argument('--backbone_lr', default=5e-6, type=_non_negative_float)
    parser.add_argument('--lr_drop', default=20, type=_positive_int)
    parser.add_argument('--weight_decay', default=1e-4, type=_non_negative_float)
    parser.add_argument('--tiling_p', default=0.2, type=_probability)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument("--bbox_loss_coef", default=1, type=_non_negative_float)
    parser.add_argument("--giou_loss_coef", default=2, type=_non_negative_float)
    parser.add_argument("--ce_loss_coef", default=2, type=_non_negative_float)
    parser.add_argument("--aux_loss_coef", default=0.5, type=_non_negative_float)
    parser.add_argument("--cost_class", default=2, type=_non_negative_float, help="Class coefficient in the matching cost")
    parser.add_argument("--cost_bbox", default=1, type=_non_negative_float, help="L1 box coefficient in the matching cost")
    parser.add_argument("--cost_giou", default=2, type=_non_negative_float, help="giou box coefficient in the matching cost")
    parser.add_argument("--focal_alpha", default=0.5, type=_probability)
    parser.add_argument("--model_name_resume_from", default='base_3_shot_softmax1', type=str)
    return parser
