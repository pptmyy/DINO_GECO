"""
test_dsgeco.py

测试目标：
1. 正确导入 src/models/DGECO.py
2. 使用 fake DINOv3Adapter / PromptEncoder / QueryGenerator / boxes_with_scores
   避免真实 DINOv3 权重、Deformable Attention、复杂后处理影响测试
3. 验证 DGECO 初始化、MLP、forward 主流程、zero-shot 分支和错误 bbox 输入
"""

import sys
import types
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
TEST_IMAGE_SIZE = 128
TEST_EMB_DIM = 32

# 让 Python 可以 import src.models.DGECO
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 让 DGECO.py 里面的绝对导入 utils.box_ops 能找到 src/utils
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================
# 2. Fake 依赖模块
# ============================================================

class FakeDINOv3Adapter(nn.Module):
    """
    伪造 DINOv3Adapter，避免测试时加载真实 DINOv3 权重。

    DGECO.forward() 中会使用：
        feats["vision_features"]
        feats["backbone_fpn"]
        feats["vision_pos_enc"]

    所以这里返回相同结构。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # DGECO.__init__ 中会读取：
        # self.dinov3_adapter.dinov3.embed_dim
        self.dinov3 = SimpleNamespace(embed_dim=TEST_EMB_DIM)

    def forward(self, x, bboxes=None):
        bs = x.shape[0]
        device = x.device
        dtype = x.dtype
        c = self.dinov3.embed_dim

        src_h = TEST_IMAGE_SIZE // 16
        l2_h = TEST_IMAGE_SIZE // 8
        l1_h = TEST_IMAGE_SIZE // 4
        src = torch.randn(bs, c, src_h, src_h, device=device, dtype=dtype)

        l1 = torch.randn(bs, c, l1_h, l1_h, device=device, dtype=dtype)
        l2 = torch.randn(bs, c, l2_h, l2_h, device=device, dtype=dtype)
        l3 = src

        pos_l1 = torch.randn(bs, c, l1_h, l1_h, device=device, dtype=dtype)
        pos_l2 = torch.randn(bs, c, l2_h, l2_h, device=device, dtype=dtype)
        pos_l3 = torch.randn(bs, c, src_h, src_h, device=device, dtype=dtype)

        return {
            "vision_features": src,
            "backbone_fpn": [l1, l2, l3],
            "vision_pos_enc": [pos_l1, pos_l2, pos_l3],
            "semantic_anchor": torch.randn(bs, c, device=device, dtype=dtype),
        }


class FakePromptEncoder(nn.Module):
    """
    伪造 PromptEncoder，只提供 get_dense_pe()。
    """

    def __init__(
        self,
        embed_dim,
        image_embedding_size,
        input_image_size,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

    def get_dense_pe(self):
        h, w = self.image_embedding_size
        return torch.zeros(1, self.embed_dim, h, w)


class FakeQueryGenerator(nn.Module):
    """
    伪造 QueryGenerator。

    不做真实 attention，只检查 DGECO 传入的关键张量 shape，
    然后直接返回 image_embeddings 作为 adapted_f / adapted_f_aux。
    """

    def __init__(
        self,
        transformer_dim,
        num_prototype_attn_steps,
        num_image_attn_steps,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_prototype_attn_steps = num_prototype_attn_steps
        self.num_image_attn_steps = num_image_attn_steps

    def forward(
        self,
        image_embeddings,
        image_pe,
        prototype_embeddings,
        hq_features,
        hq_prototypes,
        hq_pos,
        semantic_context=None,
        prototype_memory=None,
    ):
        bs, c, h, w = image_embeddings.shape

        assert c == self.transformer_dim
        assert image_pe.shape == (1, c, h, w)

        assert prototype_embeddings.ndim == 3
        assert prototype_embeddings.shape[0] == bs
        assert prototype_embeddings.shape[-1] == c

        assert isinstance(hq_features, list)
        assert len(hq_features) >= 2

        assert isinstance(hq_prototypes, list)
        assert len(hq_prototypes) == 2
        assert hq_prototypes[0].shape[0] == bs
        assert hq_prototypes[1].shape[0] == bs
        assert hq_prototypes[0].shape[-1] == c
        assert hq_prototypes[1].shape[-1] == c

        assert isinstance(hq_pos, list)
        assert len(hq_pos) >= 2
        if prototype_memory is not None:
            assert prototype_memory.shape[0] == bs
            assert prototype_memory.shape[-1] == c

        adapted_f = hq_features[0]
        adapted_f_aux = F.interpolate(hq_features[0], scale_factor=2, mode="bilinear", align_corners=False)

        return adapted_f, adapted_f_aux


def fake_boxes_with_scores(centerness, outputs_coord, sort=False, validate=False):
    """
    伪造 boxes_with_scores。

    DGECO.forward() 只要求它返回：
        outputs, ref_points
    """
    bs = centerness.shape[0]
    device = centerness.device

    outputs = []
    for _ in range(bs):
        outputs.append(
            {
                "boxes": torch.zeros(1, 4, device=device),
                "box_v": torch.ones(1, device=device),
                "scores": torch.ones(1, device=device),
            }
        )

    ref_points = torch.zeros(bs, 1, 2, device=device)
    return outputs, ref_points


# ============================================================
# 3. 在 import DGECO.py 之前注入 fake 模块
# ============================================================

def install_fake_modules():
    """
    在导入 src.models.DGECO 之前，把 DGECO.py 中依赖的重模块替换成 fake 模块。

    这样可以避免：
    1. 真实 DINOv3Adapter 加载失败
    2. QueryGenerator 中依赖 Deformable Attention 编译算子
    3. PromptEncoder 构造参数不一致
    4. boxes_with_scores 真实后处理影响 smoke test
    """

    # fake src.models.backbones.dinov3_adapter
    fake_dino_module = types.ModuleType("src.models.backbones.dinov3_adapter")
    fake_dino_module.DINOv3Adapter = FakeDINOv3Adapter
    sys.modules["src.models.backbones.dinov3_adapter"] = fake_dino_module

    # fake src.models.query_generator
    fake_query_module = types.ModuleType("src.models.query_generator")
    fake_query_module.QueryGenerator = FakeQueryGenerator
    sys.modules["src.models.query_generator"] = fake_query_module

    # fake src.models.prompt_encoder
    fake_prompt_module = types.ModuleType("src.models.prompt_encoder")
    fake_prompt_module.PromptEncoder = FakePromptEncoder
    sys.modules["src.models.prompt_encoder"] = fake_prompt_module

    # fake utils.box_ops
    # 你的 DGECO.py 里写的是：
    # from utils.box_ops import boxes_with_scores
    fake_utils_module = types.ModuleType("utils")
    fake_box_ops_module = types.ModuleType("utils.box_ops")
    fake_box_ops_module.boxes_with_scores = fake_boxes_with_scores

    sys.modules["utils"] = fake_utils_module
    sys.modules["utils.box_ops"] = fake_box_ops_module
    sys.modules["src.utils.box_ops"] = fake_box_ops_module


@pytest.fixture()
def dgeco_mod(monkeypatch):
    """
    导入 src.models.DGECO，并替换必要依赖。
    """

    install_fake_modules()

    # 确保每次测试都重新导入 DGECO.py，避免缓存旧模块
    sys.modules.pop("src.models.DGECO", None)

    dgeco_module = importlib.import_module("src.models.DGECO")

    # DGECO.py 中 MLP.forward 使用 F.relu。
    # 如果源码还没加 from torch.nn import functional as F，这里临时补上。
    monkeypatch.setattr(dgeco_module, "F", F, raising=False)

    return dgeco_module


# ============================================================
# 4. 构造测试模型
# ============================================================

def build_test_model(dgeco_module, training=True, zero_shot=False):
    return dgeco_module.DGECO(
        image_size=TEST_IMAGE_SIZE,
        dinov3_model_size="base",
        dinov3_patch_size=16,
        dinov3_out_feature_indexes=(2, 5, 8, 11),
        dinov3_freeze_encoder=True,
        dinov3_pretrained_weights=None,
        dinov3_conv_inplane=16,
        num_objects=2,
        kernel_dim=1,
        zero_shot=zero_shot,
        training=training,
        reduction=16,
        query_output_stride=4,
        stride2_refinement=True,
        center_gaussian_head=True,
        num_prototypes=4,
        mutual_adapter_layers=1,
        decoupled_heads=True,
    )


def make_valid_inputs(bs=2):
    x = torch.randn(bs, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)

    bboxes = torch.tensor(
        [
            [[10.0, 10.0, 30.0, 32.0], [50.0, 48.0, 78.0, 82.0]],
            [[12.0, 14.0, 34.0, 36.0], [70.0, 72.0, 105.0, 110.0]],
        ],
        dtype=torch.float32,
    )

    return x, bboxes


# ============================================================
# 5. 测试用例
# ============================================================

def test_mlp_forward_and_backward(dgeco_mod):
    """
    单独测试 MLP：
    1. 输出 shape 是否正确
    2. 是否可以反向传播
    """
    mlp = dgeco_mod.MLP(
        input_dim=32,
        hidden_dim=64,
        output_dim=4,
        num_layers=3,
    )

    x = torch.randn(2, 10, 32, requires_grad=True)
    y = mlp(x)

    assert y.shape == (2, 10, 4)
    assert torch.isfinite(y).all()

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dgeco_init_success(dgeco_mod):
    """
    测试 DGECO 初始化是否成功。
    """
    model = build_test_model(dgeco_mod, training=True)

    assert hasattr(model, "dinov3_adapter")
    assert hasattr(model, "dino_prompt_encoder")
    assert hasattr(model, "adapt_features")

    assert model.emb_dim == 32
    assert model.image_size == TEST_IMAGE_SIZE
    assert model.reduction == 16
    assert model.num_objects == 2
    assert model.pretrain is False


def test_dgeco_forward_train_smoke(dgeco_mod):
    """
    测试 DGECO.forward() 在 training=True 下能否完整跑通。

    覆盖：
    1. DINOv3Adapter 输出 feats
    2. roi_align 提取 exemplars
    3. shape_or_objectness 生成 shape embedding
    4. prototype_embeddings 拼接
    5. adapt_features 调用
    6. class_embed / bbox_embed 输出
    7. boxes_with_scores 后处理
    """
    torch.manual_seed(0)

    model = build_test_model(dgeco_mod, training=True)
    model.train()

    x, bboxes = make_valid_inputs(bs=2)

    out = model(x, bboxes)

    assert isinstance(out, tuple)
    assert len(out) == 5

    outputs, ref_points, centerness, outputs_coord, aux = out

    bs = x.shape[0]

    assert isinstance(outputs, list)
    assert len(outputs) == bs
    assert ref_points.shape == (bs, 1, 2)

    assert centerness.shape == (bs, 1, TEST_IMAGE_SIZE // 4, TEST_IMAGE_SIZE // 4)
    assert outputs_coord.shape == (bs, 4, TEST_IMAGE_SIZE // 4, TEST_IMAGE_SIZE // 4)

    assert torch.isfinite(centerness).all()
    assert torch.isfinite(outputs_coord).all()

    assert isinstance(aux, dict)
    outputs_aux = aux["refine_outputs"]
    ref_points_aux = aux["refine_ref_points"]
    centerness_aux = aux["refine_centerness"]
    outputs_coord_aux = aux["refine_boxes"]
    center_heatmap = aux["center_heatmap"]

    assert isinstance(outputs_aux, list)
    assert len(outputs_aux) == bs
    assert ref_points_aux.shape == (bs, 1, 2)
    assert centerness_aux.shape == (bs, 1, TEST_IMAGE_SIZE // 2, TEST_IMAGE_SIZE // 2)
    assert outputs_coord_aux.shape == (bs, 4, TEST_IMAGE_SIZE // 2, TEST_IMAGE_SIZE // 2)
    assert center_heatmap.shape == (bs, 1, TEST_IMAGE_SIZE // 2, TEST_IMAGE_SIZE // 2)

    for item in outputs:
        assert "boxes" in item
        assert "box_v" in item
        assert "scores" in item


def test_dgeco_forward_zero_shot_uses_self_num_objects(dgeco_mod):
    """
    测试 zero_shot=True 时，num_objects 使用 self.num_objects。
    """
    torch.manual_seed(0)

    model = build_test_model(dgeco_mod, training=True, zero_shot=True)
    model.train()

    x, bboxes = make_valid_inputs(bs=2)

    out = model(x, bboxes)

    outputs, ref_points, centerness, outputs_coord, aux = out

    assert len(outputs) == 2
    assert ref_points.shape == (2, 1, 2)
    assert centerness.shape == (2, 1, TEST_IMAGE_SIZE // 4, TEST_IMAGE_SIZE // 4)
    assert outputs_coord.shape == (2, 4, TEST_IMAGE_SIZE // 4, TEST_IMAGE_SIZE // 4)


def test_dgeco_forward_exports_verification_features_when_enabled(dgeco_mod):
    torch.manual_seed(0)

    model = build_test_model(dgeco_mod, training=True)
    model.return_candidate_features = True
    model.train()

    x, bboxes = make_valid_inputs(bs=2)
    outputs, _, _, _, _ = model(x, bboxes)

    for item in outputs:
        assert "candidate_features" in item
        assert "exemplar_features" in item
        assert item["candidate_features"].ndim == 2
        assert item["candidate_features"].shape[-1] == model.emb_dim
        assert item["exemplar_features"].ndim == 2
        assert item["exemplar_features"].shape[-1] == model.emb_dim


def test_dgeco_forward_bad_bbox_shape_should_fail(dgeco_mod):
    """
    正确 bboxes 应该是：
        [B, num_objects, 4]

    这里故意传 [B, 4]，应该报错。
    """
    model = build_test_model(dgeco_mod, training=True)
    model.train()

    x = torch.randn(2, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)

    bad_bboxes = torch.tensor(
        [
            [10.0, 10.0, 30.0, 32.0],
            [12.0, 14.0, 34.0, 36.0],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(Exception):
        _ = model(x, bad_bboxes)


def test_center_gaussian_loss_uses_heatmap_resolution():
    from src.utils.losses import SetCriterion

    criterion = SetCriterion(
        0,
        matcher=None,
        weight_dict={"loss_center_gaussian": 1.0},
        losses=[],
        center_gaussian_sigma=2.0,
    )
    heatmap = torch.zeros(1, TEST_IMAGE_SIZE // 2, TEST_IMAGE_SIZE // 2, requires_grad=True)
    targets = [
        {
            "boxes": torch.tensor([[0.25, 0.25, 0.50, 0.50]], dtype=torch.float32),
            "labels": torch.zeros(1, dtype=torch.long),
        }
    ]

    losses = criterion.center_gaussian_loss(targets, heatmap)

    assert torch.isfinite(losses["loss_center_gaussian"])
    assert losses["center_gaussian_positive_sum"].item() > 0
    losses["loss_center_gaussian"].backward()
    assert heatmap.grad is not None
    assert torch.isfinite(heatmap.grad).all()
