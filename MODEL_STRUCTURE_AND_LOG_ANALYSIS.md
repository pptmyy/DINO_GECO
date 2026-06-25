# DINO-SAM-GECO2 深度学习网络结构与日志分析

生成日期：2026-06-24

本文基于当前工作区代码、`logs/log/*/metrics.jsonl`、`outputs/eval_runs/*`、`eval_runs/*` 与 `sweep_outputs/*` 中的本地结果整理。重点总结当前项目的深度学习网络结构，并分析训练日志和输出评估结果的主要变化。

## 1. 项目定位

DINO-SAM-GECO2 面向 FSC147 少样本目标计数与检测。模型输入一张图像和少量 exemplar boxes，先用 DINOv3 提取密集视觉特征，再把 exemplar 区域特征和几何信息变成类别原型，最后在网格位置上预测候选目标中心分数与归一化边界框。

核心代码位置：

- 主模型：`src/models/DGECO.py`
- DINOv3 适配器：`src/models/backbones/dinov3_adapter.py`
- exemplar 条件查询生成器：`src/models/query_generator.py`
- 多尺度查询聚合：`src/models/scale_query_aggregator.py`
- 匈牙利匹配与损失：`src/models/matcher.py`、`src/utils/losses.py`
- 后处理：`src/utils/postprocess.py`
- SAM3 可选掩码生成：`infer.py`

## 2. 网络整体结构

整体数据流如下：

```text
image [B,3,H,W] + exemplar boxes [B,N,4]
        |
        v
DINOv3Adapter
  - frozen DINOv3 ViT backbone
  - CNN spatial prior branch
  - semantic/detail feature fusion
        |
        +--> vision_features c3: [B,C,H/16,W/16]
        +--> backbone_fpn: c1 [H/4], c2 [H/8], c3 [H/16]
        +--> vision_pos_enc: c1/c2/c3 positional encodings
        +--> semantic_anchor: DINOv3 class token
        |
        v
ROI Align exemplar regions on c1/c2/c3
        |
        v
exemplar visual tokens + 9D geometry embedding
        |
        v
QueryGenerator
  - prototype cross-attention, 3 steps
  - MSDeformAttn image attention, 2 steps
  - scale-aware aggregation c3 -> c2 -> c1
  - optional semantic anchor fusion
        |
        v
dense query feature map, default output_stride=8 in latest runs
        |
        v
Detection heads
  - class_embed: centerness/objectness logit
  - bbox_embed: normalized box offsets
  - auxiliary class/bbox heads
        |
        v
boxes_with_scores -> candidate boxes
        |
        v
postprocess: threshold + NMS + optional verification/SAM3 mask
```

### 2.1 DINOv3Adapter

`DINOv3Adapter` 是视觉主干适配层。当前默认配置使用 DINOv3 ViT-Base，`embed_dim=768`，patch size 为 16，默认冻结 DINOv3 编码器，只训练外部适配层和检测相关模块。

结构要点：

- DINOv3 提取多层 token，默认中间层索引来自 `DGECO` 构造参数 `(2, 5, 8, 11)`。
- token reshape 为语义特征图，再插值到多个尺度。
- `SpatialPriorModulev2` 从原图走轻量 CNN 分支，输出 c1/c2/c3/c4 细节特征。
- 每个尺度将 DINOv3 语义特征与 CNN 细节特征 concat，再用 1x1 conv + GroupNorm 融合到统一通道数。
- 输出 `backbone_fpn=[c1,c2,c3]`，其中 `c1` 约为 H/4，`c2` 约为 H/8，`c3` 约为 H/16。
- `semantic_anchor` 取最后一个 DINOv3 class token，供最新语义锚点版本使用。

### 2.2 Exemplar 原型构造

主模型 `DGECO.forward` 会用 `roi_align` 在主特征 `vision_features` 以及高分辨率 `c1/c2` 上裁剪 exemplar 区域。当前默认 `kernel_dim=1`，因此每个 exemplar 在每个尺度上通常形成 1 个视觉 token。

除视觉 token 外，模型还为每个 exemplar box 构造 9 维几何特征：

- box 宽、高、面积相对 image size 的比例；
- 宽高比与高宽比；
- 宽、高、面积的 log scale；
- 面积开方后的尺度。

这 9 维几何特征经过两层 MLP 映射到 `embed_dim`，再与视觉 ROI token 拼接，形成最终 `prototype_embeddings`。这一步让模型不仅知道 exemplar 的外观，也知道 exemplar 的尺度与形状先验。

### 2.3 QueryGenerator

`QueryGenerator` 将 dense image feature 调整为目标类别相关的查询特征。它同时处理主尺度 c3 和高分辨率 c1/c2。

主要模块：

- 每个尺度有 3 个 `PrototypeAttentionBlock`，使用 image tokens 作为 query，exemplar prototype tokens 作为 key/value，完成 exemplar 条件特征注入。
- 每个尺度有 2 个 `MSDeformAttn`，对图像 token 做局部/稀疏的空间上下文聚合。
- 之后进入 `ScaleAwareQueryAggregator` 做尺度融合。

这种结构本质上是“先用 exemplar 原型让图像特征变成类别相关，再用可变形注意力补空间上下文，最后从语义尺度向细节尺度逐步融合”。

### 2.4 多尺度查询聚合

`ScaleAwareQueryAggregator` 的输入为：

- `q3`: 主语义查询，来自 c3/H16；
- `q2`: 中尺度查询，来自 c2/H8；
- `q1`: 高分辨率查询，来自 c1/H4；
- exemplar prototype context；
- 可选 DINOv3 `semantic_anchor`。

融合过程：

- `q3` 上采样到 `q2` 分辨率，与 `q2` 通过 `ScaleFusionBlock` 融合；
- 融合后的 `q2` 再上采样到 `q1` 分辨率，与 `q1` 融合；
- `ScaleFusionBlock` 同时使用空间 gate 和通道 gate，让 prototype/semantic context 控制高层语义向局部细节的注入强度；
- `output_stride` 可为 2、4、8。最新 2026-06-18/2026-06-20 长训练使用 `query_output_stride=8`。

### 2.5 检测头与候选框生成

检测头非常轻：

- `class_embed`: `Linear(embed_dim, 1)`，输出 centerness/objectness logit；
- `bbox_embed`: 3 层 MLP，输出 4 个归一化偏移量；
- 非 pretrain 模式下还有一组 auxiliary class/bbox head。

`boxes_with_scores` 会对 centerness map 做 sigmoid 和 3x3 max pooling，保留局部峰值点，然后用预测偏移量还原为归一化 `xyxy` box。训练时阈值使用 median，验证时阈值使用 max/8。

验证模式下，`Box_correction` 会调用 `MaskProcessor` 修正框，并将 mask IoU 作为 scores。单图或数据集推理时，`infer.py` 还支持可选 SAM3 mask refinement。

### 2.6 训练损失

训练环节使用 `PointLossHungarianMatcher` 做单图候选框到 GT box 的 Hungarian 匹配。匹配 cost 由三部分组成：

- 分类/objectness cost；
- L1 bbox cost；
- GIoU cost。

`SetCriterion` 中的损失包括：

- `loss_ce`: quality focal loss，目标质量来自匹配框 IoU，并对 GT 中心邻域补充监督；
- `loss_bbox`: 匹配框与 GT 的 L1 loss；
- `loss_giou`: GIoU loss；
- auxiliary branch loss 由训练脚本加权合并。

当前默认权重在 `arg_parser.py` 中为 `ce_loss_coef=2`、`bbox_loss_coef=1`、`giou_loss_coef=2`、`aux_loss_coef=0.5`。

### 2.7 后处理

`filter_detections` 负责推理/评估后处理，支持：

- `static_ratio`、`quantile`、`regime_adaptive` 阈值模式；
- hard NMS、soft NMS、`dense_soft`；
- min/max box area 过滤；
- adaptive dense/sparse regime；
- optional verification，例如 exemplar geometry 或 feature similarity。

最新默认参数倾向：

- `score_threshold=0.20`
- `score_ratio=0.50`
- `nms_iou=0.30`
- `nms_method=dense_soft`
- dense regime 使用更低分数比例和更高 NMS IoU，以增加密集场景保留。

## 3. 关键训练日志变化

下表来自 `logs/log/*/metrics.jsonl`。早期日志没有记录 detection F1/AP，2026-06-18 之后新增了检测指标、候选数量统计、dense/sparse 后处理统计等。

| Run | 有效 epoch | 主要设置/变化 | 最佳 Val MAE | 最佳 Val RMSE | 最佳 Val F1@IoU50 | 备注 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `DGECO2FSCD_20260523_171004` | 31 | 早期版本，`score_threshold=0.05`、`score_ratio=0.25`、`nms_iou=0.5` | 112.55 | 158.59 | N/A | 计数误差很高，训练尚未稳定 |
| `DGECO2FSCD_20260525_200135` | 63 | 后处理阈值提高到 `0.1/0.4` | 41.22 | 78.72 | N/A | 相比 5/23 大幅下降，但仍明显欠佳 |
| `DGECO2FSCD_20260602_101741` | 57 | `lr_drop=20`，`score_threshold=0.2`、`score_ratio=0.5`、`nms_iou=0.3` | 20.70 | 49.56 | N/A | 计数性能第一次进入较好区间 |
| `DGECO2FSCD_20260608_163211` | 102 | P3 版本，加入更完整后处理参数 | 21.12 | 48.88 | N/A | 当前计数 RMSE 最强主线之一 |
| `DGECO2FSCD_20260618_163845` | 50 | `query_output_stride=8`，`dense_soft`，无 semantic anchor | 24.00 | 66.37 | 0.4755 | 检测 F1 明显可观，但计数 RMSE 退化 |
| `DGECO2FSCD_20260620_135659` | 59 | 在 6/18 基础上打开 `use_semantic_anchor=True` | 21.52 | 60.66 | 0.5085 | semantic anchor 提升 F1/MAE，但 RMSE 仍弱于 P3 |

### 3.1 训练趋势结论

1. 5 月底到 6 月初，主要收益来自训练/后处理稳定化。Val RMSE 从 158.59 降到 49.56，Val MAE 从 112.55 降到 20.70。
2. 2026-06-08 的 P3 版本在计数上最均衡，best RMSE 约 48.88，best MAE 约 21.12。
3. 2026-06-18 的 stride8 版本引入了检测指标和更完整的候选统计，F1@IoU50 最高约 0.4755，但 Val RMSE 升到约 66。
4. 2026-06-20 的 semantic anchor 版本将 F1@IoU50 推到约 0.5085，Val MAE 也恢复到 21.52，但 Val RMSE 仍为 60.66，说明大误差样本仍没有解决。
5. 最新 P3A/semantic 的候选数量从训练初期明显下降：Val `preds_per_gt` 从 1.66 下降到 0.92，`val_pred_error` 从 +53024 变为 -6586，说明模型从早期过检转向略欠检。

## 4. 输出评估结果变化

下表来自 `outputs/eval_runs/*/summary.json`、`eval_runs/*/summary.json` 和 sweep summary。

| 输出目录 | Val MAE | Val RMSE | Signed Error | Pred Avg | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `outputs/eval_runs/baseline_epoch47_val` | 20.70 | 49.58 | -5.28 | 56.90 | epoch47 基线，计数表现稳定 |
| `outputs/eval_runs/p3_20260608_best_mae_val` | 21.12 | 49.46 | -4.38 | 57.80 | P3 best MAE |
| `outputs/eval_runs/p3_20260608_best_rmse_val` | 21.21 | 48.88 | -1.44 | 60.73 | P3 best RMSE，整体 RMSE 最好之一 |
| `outputs/eval_runs/p3_20260608_last_val` | 21.29 | 49.17 | -2.50 | 59.67 | P3 last checkpoint，接近 best |
| `outputs/eval_runs/sweep_postprocess_v2_val` best RMSE | 20.61 | 48.60 | -6.52 | 55.65 | 后处理 sweep 的 RMSE 最优组合 |
| `outputs/eval_runs/p3_20260608_verification_sweep` best RMSE | 21.16 | 48.83 | -1.49 | N/A | feature similarity 验证对 RMSE 有轻微改善 |
| `outputs/eval_runs/stride8_20260618_best_rmse_val` | 24.30 | 66.37 | -4.06 | 58.11 | stride8 无 semantic anchor，计数退化 |
| `outputs/eval_runs/stride8_20260618_best_f1_val` | 24.00 | 66.94 | -1.39 | 60.79 | 检测 F1 checkpoint，计数不是最优 |
| `eval_runs/p3a_20260620_stride8_semantic/best_val_rmse_val` | 21.52 | 60.67 | -5.14 | 57.04 | semantic anchor 改善 6/18，但高计数 RMSE 仍大 |
| `eval_runs/p3a_20260620_stride8_semantic/best_val_f1_val` | 21.92 | 61.77 | -5.07 | 57.10 | F1 checkpoint 与 RMSE checkpoint 接近 |

### 4.1 后处理 sweep 观察

`sweep_postprocess_v2_val` 在 288 个组合中：

- best MAE：MAE 20.53，RMSE 50.35，组合为 `score_threshold=0.25`、`score_ratio=0.5`、`nms_iou=0.25`、`static_ratio`。
- best RMSE：MAE 20.61，RMSE 48.60，组合为 `score_threshold=0.25`、`score_ratio=0.5`、`nms_iou=0.25`、`regime_adaptive`，dense candidate threshold 为 256。

因此，如果目标是验证集计数 RMSE，当前本地输出里最强的是 P3 checkpoint + sweep 后处理，而不是最新 semantic anchor checkpoint。

## 5. P3 与 P3A/semantic 候选生成对比

`eval_runs/candidate_generation_diagnostics/p3_vs_latest/diagnostic_report.md` 比较了：

- Baseline：`p3_20260608_best_rmse`
- Candidate：`p3a_20260620_stride8_semantic_best_rmse`

整体变化：

| 指标 | P3 baseline | P3A semantic | Delta |
| --- | ---: | ---: | ---: |
| MAE | 21.21 | 21.52 | +0.31 |
| RMSE | 48.88 | 60.67 | +11.79 |
| Signed Error | -1.44 | -5.14 | -3.70 |
| Pre-NMS / GT | 1.34 | 1.03 | -0.32 |
| Post-NMS / GT | 0.98 | 0.92 | -0.06 |
| Score Mean | 0.168 | 0.310 | +0.142 |
| Effective Threshold | 0.367 | 0.419 | +0.052 |

解释：

- P3A/semantic 的分数整体更高，阈值也更高，最终候选更少。
- 低计数段误检明显减少，MAE 下降。
- 高计数段候选生成不足，导致大目标数图片明显欠计数，RMSE 被拉高。

分桶变化：

| Count Bin | P3 MAE/RMSE | P3A MAE/RMSE | 变化 |
| --- | ---: | ---: | --- |
| 7-20 | 8.89 / 18.89 | 5.48 / 17.48 | 明显改善，过计数减少 |
| 20-50 | 11.95 / 23.77 | 10.36 / 22.25 | 改善，预测均值更接近 GT |
| 50-100 | 25.97 / 41.41 | 27.77 / 42.90 | 小幅退化 |
| 100-300 | 48.67 / 62.14 | 51.35 / 67.60 | 退化，但 signed error 从 -25.24 改到 -17.82 |
| >300 | 145.22 / 215.98 | 199.70 / 298.68 | 严重退化，主要 RMSE 来源 |

高计数瓶颈不是 NMS 过强，而是候选生成阶段已经不足。诊断报告中 `>300` bin 的 P3A `pre_nms_per_gt=0.756`、`post_nms_per_gt=0.756`，且 NMS keep ratio 接近 1，说明进入 NMS 前候选已经偏少。

典型大误差图像：

- `935.jpg`: GT 1599，P3 pred 1029，P3A pred 661。
- `7656.jpg`: GT 1229，P3 pred 620，P3A pred 391。
- `1956.jpg`: GT 1232，P3 pred 1216，P3A pred 700。
- `949.jpg`: GT 1123，P3 pred 1058，P3A pred 615。

这些样本表明最新 P3A/semantic 版本在超高密度目标场景出现候选召回不足。

## 6. 当前模型优缺点

优点：

- DINOv3 冻结主干提供稳定语义表示，训练成本主要集中在 adapter、query generator 和 detection heads。
- exemplar visual token + geometry token 的原型设计适合少样本计数，能同时利用外观和尺度信息。
- 多尺度 query aggregation 能让高层语义与低层细节融合，对小目标和低计数误检有帮助。
- 最新日志已经包含检测 F1/AP、候选数量、dense/sparse 后处理比例等诊断信息，便于定位问题。

主要问题：

- P3A/semantic 对低计数段更保守，但对高密度图像召回不足，RMSE 被少数高 count 图片显著拉高。
- `output_stride=8` 降低了候选网格密度，可能是高密度场景候选不足的重要原因之一。
- semantic anchor 提高了目标相关性和 F1，但也可能强化了高阈值/少候选倾向。
- 最新版本检测 F1 与计数 RMSE 不完全一致，说明框质量提升并不等价于计数最优。

## 7. 实验结论

如果当前目标是 FSC147 验证集计数：

- 优先参考 `outputs/eval_runs/sweep_postprocess_v2_val` 的 best RMSE 组合，当前本地最优 RMSE 约 48.60。
- P3 checkpoint 的计数结果整体优于 6/18、6/20 的 stride8/P3A 版本。

如果当前目标是目标检测框质量：

- 6/20 semantic anchor 版本的 F1@IoU50 最高，约 0.5085，比 6/18 的约 0.4755 更好。
- 但它在高计数场景候选不足，不能直接作为计数最优模型。

后续如果继续优化，建议优先验证：

1. 对高计数图像使用更密的候选输出，例如重新比较 `query_output_stride=4/8`。
2. 对 dense regime 单独降低阈值或提高 candidate cap，避免 `>300` 图像候选不足。
3. 保留 semantic anchor，但给高密度样本引入 count-aware 或 density-aware 后处理策略。
4. 在训练日志中继续同时保存 count RMSE/MAE 与 detection F1/AP，避免只优化检测指标导致计数退化。

