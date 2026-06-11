# 第四版代码修改建议与后续优化方向

更新时间：2026-06-08

本文基于当前工作区代码、`logs/log/**`、`outputs/eval_runs/**`、`outputs/vis_val_3_args/**`，以及截至 2026-06-08 可查的公开资料重写。旧文档中重复的历史记录、过长的 batch 字段逐项解释、已经完成的待办描述和过时调参建议已删除。

## 当前结论

当前项目已经从早期 density/count 混合监督转为 detection-only 主线：DINOv3 backbone + exemplar-conditioned dense query + box/objectness head，训练通过 Hungarian matching 的 focal CE、bbox L1、GIoU，推理通过阈值、NMS 和可选 verification 后的框数量计数。

需要明确区分两类证据：

- 已有有效基线：`logs/log/DGECO2FSCD_20260602_101741`、`outputs/eval_runs/baseline_epoch47_val`、`outputs/eval_runs/sweep_postprocess_v2_val`。这些结果主要验证旧 epoch 47 checkpoint 与后处理扫描。
- 尚未出现的证据：第四版当前代码，尤其是 `P3 ScaleAwareQueryAggregator` 落地后的第一次完整训练/验证/测试 log。现有日志只到旧 run 的 epoch 58 batch 400，不能证明 P3 是否有效。

因此，下一步不是继续叠加新结构，而是先用第四版当前代码完成一次干净首测，并把日志、`metrics.jsonl`、per-image eval、count-bin summary 和 top-error 文件保存完整。

## 2026-06-08 外部资料核对

截至 2026-06-08，外部资料对本项目的直接启发如下：

- Meta DINOv3 官方资料强调 frozen backbone 的高分辨率 dense feature 和 lightweight adapter 价值，这支持当前“冻结 DINOv3、训练 adapter/query/head”的路线，不支持立刻全量微调 backbone。
- GECO2 论文提出 gradual cross-scale dense query aggregation，目标正是 dense small object 和尺度变化；当前 P3 最小版与这个方向一致，但还未完成第四版训练验证。
- DAVE 的 detect-and-verify 范式强调先生成高召回候选，再做 outlier verification；这支持 P2 作为校准/过滤层，但也说明当 `preds_before_nms < gt_count` 时，verification 不能修复候选召回不足。
- SAM 3 已从单纯 promptable segmentation 扩展到概念级检测、分割、跟踪，并支持 exemplar prompt；本项目可把 SAM3 作为 exemplar mask prior，但不应直接把伪 mask 当作全量 GT。
- RF-DETR 已进入 Hugging Face Transformers 主线文档，核心包含 DINOv2 backbone、多尺度特征、浅层 DETR decoder 和 mixed-query；可作为 P3-B top-K 候选精修参考，而不是直接替换当前 dense detector。
- DQ-DETR 针对 tiny object 提出 counting-guided feature enhancement 和 dynamic query selection；本项目可借鉴“动态候选/密度先验”，但当前架构不是标准 fixed-query DETR，不能直接照搬。
- PyTorch 官方页面已列出 2.10/2.11 + CUDA 13.0 wheel。当前 `requirements.txt` 固定 `torch==2.10.0+cu130` 是可解释的，但第四版首测不要同时升级 PyTorch，避免把依赖变化和结构收益混在一起。
- Muon optimizer 的公开论文定位是矩阵正交化优化器，主要证明大规模语言模型训练可扩展；在本项目里只适合作为后期 optimizer ablation，不能作为解决高计数漏检的主方案。

## 已完成事项

| 编号 | 状态 | 内容 | 本地证据 |
|---|---|---|---|
| P0-A | 已完成 | 推理/评估 GT count 口径修正，`gt_count` 使用有效 `gt_bboxes` 数量，`density_map.sum()` 只保留为 `density_sum_debug`。 | `infer.py`、`tools/eval_checkpoint.py` |
| P0-B | 已完成 | 单 checkpoint 评估工具落地，可输出 `summary.json`、`summary_by_count_bin.csv`、`top_abs_errors.csv` 和 per-image jsonl。 | `tools/eval_checkpoint.py`、`tools/diagnostics.py` |
| P0-C | 已完成 | 第四版配置固化，关键后处理/验证参数进入配置和脚本。 | `configs/fourth_fsc147_epoch47.json`、`configs/linux_fsc147.json`、`scripts/run_train_linux_new.sh` |
| P1 | 已完成 | 后处理工程化：`static_ratio`、`quantile`、`regime_adaptive`、面积过滤、NMS 前后统计、有效阈值和原始候选索引返回。 | `src/utils/postprocess.py` |
| P1 sweep | 已完成 | epoch 47 checkpoint 的 288 组合后处理扫描完成。 | `outputs/eval_runs/sweep_postprocess_v2_val/summary.json` |
| P2 代码 | 已完成 | 轻量 verification 支持 `none`、`exemplar_geometry`、`feature_similarity`，支持 `hard`、`soft`、`sparse_hard`。 | `src/utils/verification.py`、`train.py`、`infer.py` |
| P3 最小版 | 已完成代码，待首测 | `ScaleAwareQueryAggregator` 已接入 `QueryGenerator`，执行 `q3 -> q2 -> q1 -> q_out` 渐进聚合。 | `src/models/scale_query_aggregator.py`、`src/models/query_generator.py` |
| P5 核心诊断 | 已完成代码，待首测日志验证 | 训练日志新增 `pre_nms`、`post_nms`、`post_verify`、`thr`。评估统计包含 count bin、over/under、NMS 前后数量。 | `train.py`、`tools/diagnostics.py` |
| 测试 | 已完成 | 2026-06-08 复测通过。 | `conda run -n ScientificResearch python -m pytest -q test_postprocess_verification.py test_dsgeco.py`，结果 `15 passed` |

## 已完成但必须等第四版首测日志确认的部分

这些内容已经写进代码，但现在不能下结论：

| 项目 | 等待什么证据 | 判断方式 |
|---|---|---|
| P3 最小结构是否有效 | 新 run 的 `metrics.jsonl`、`train.log`、val/test eval 输出 | 对比旧 epoch 47：`100-300`、`>300` 的 `avg_pre_nms / gt_avg` 是否上升，signed error 是否接近 0，`7-20` 过检是否没有反弹 |
| P5 新日志是否足够 | 第四版训练日志中的 `pre_nms/post_nms/post_verify/thr` | 能否定位每个 epoch 是候选召回不足、NMS 压制、verification 压制还是阈值偏移 |
| 第四版是否应该提前停止 | 第四版完整 val/test 指标趋势 | 不能沿用旧 run 的 epoch 47 早停结论，必须看新结构的 best epoch 是否后移 |
| P1 `regime_adaptive` 是否应成为默认 | 第四版新 checkpoint 的后处理 sweep | 旧 checkpoint 的 combo 282 只能作为参考，不能直接替代新结构默认值 |

## 未完成事项

| 编号 | 当前状态 | 为什么暂不继续改 |
|---|---|---|
| P2 verification sweep | 未完成 | 代码已有，但还没有 `feature_similarity + soft/sparse_hard` 的系统扫描。可在旧 checkpoint 上补跑，也应在第四版首测 checkpoint 后重跑。 |
| P3-A DINOv3-guided query | 未完成 | `DINOv3Adapter` 当前计算了 `c4` 和 class token，但返回值只给 `c1/c2/c3`，class token 没进入 gate/query/score。必须先看 P3 最小版是否有效。 |
| 多层深监督 | 未完成 | 只有主输出和现有 aux 输出，尚未对 `q3/q2/q1/q_out` 分层监督。需等 P3 是否平台期过早后再决定。 |
| DINO feature consistency loss | 未完成 | 目前没有 evidence 证明 query/fusion 层破坏 frozen DINO dense feature。等首测日志和 per-image 误差再决定。 |
| P2-B count-bin prior / density-aware gate | 未完成 | 当前低计数假阳性存在，但应先确认 P3 是否改善高计数召回；监督只能来自有效 `gt_bboxes` count，不能来自 density sum。 |
| P3-B mixed-query refiner | 未完成 | 适合修正高分假阳性和框质量，不适合第一步修复 `preds_before_nms < gt_count`。 |
| P4 SAM3 mask-weighted pooling | 未完成 | 项目已有 SAM3 伪实例文件，但 dataset 和模型没有可选 mask pooling；需先做 mask 质量统计。 |
| P6 Muon optimizer | 未完成 | 只适合作为结构固定后的 optimizer ablation，不应和 P3-A/P2-B/P3-B 同轮叠加。 |
| P7 RL / policy optimization | 未完成，后期候选 | 只适合作为后处理策略学习或 metric-aware objectness 重加权，不应作为当前 P3 首测阶段的主线。必须先完成 P3 checkpoint 评估、count-bin 诊断和 P2 verification sweep。 |

## 现有基线结果

最新可用旧训练日志：

```text
logs/log/DGECO2FSCD_20260602_101741
完整汇总到 epoch 57
epoch 58 只有 batch 级日志到 batch 400/1829
无 Traceback / RuntimeError / OOM / NaN / Inf
```

旧 run 最优点：

```text
epoch 47:
val_mae=20.703
val_rmse=49.559
test_mae=19.030
test_rmse=96.710
```

`outputs/eval_runs/baseline_epoch47_val`：

```text
overall: MAE=20.698, RMSE=49.578, signed=-5.275
over=515, under=685, equal=86
avg_pre_nms=78.680, avg_post_nms=56.918, avg_effective_thr=0.374
```

count-bin 结构：

```text
7-20:    MAE=6.885,   signed=+5.095
20-50:   MAE=10.737,  signed=+0.085
50-100:  MAE=24.378,  signed=-5.319
100-300: MAE=51.613,  signed=-30.877
>300:    MAE=163.216, signed=-93.757
```

结论：低计数图有稳定过检，高计数密集图明显欠检。后处理不能单独解决两头问题。

`outputs/eval_runs/sweep_postprocess_v2_val`：

```text
baseline combo 13:
  static_ratio, score_threshold=0.20, score_ratio=0.50, nms_iou=0.30
  MAE=20.698, RMSE=49.578

best_by_mae combo 30:
  MAE=20.534, RMSE=50.346
  但 >300 signed 恶化到 -127.730

best_by_rmse combo 282:
  regime_adaptive, score_threshold=0.25, score_ratio=0.50, nms_iou=0.25,
  adaptive_dense_candidate_threshold=256
  MAE=20.608, RMSE=48.601
```

因此 combo 282 只能作为旧 checkpoint 的 RMSE 导向参考，不应直接作为第四版默认配置。

## 第四版首测必须产出的文件

建议下一次完整训练/测试至少保留：

```text
logs/log/<new_fourth_run>/args.json
logs/log/<new_fourth_run>/train.log
logs/log/<new_fourth_run>/metrics.jsonl
checkpoints/<run_name>/<run_name>_best_val_mae.pth
checkpoints/<run_name>/<run_name>_best_val_rmse.pth
checkpoints/<run_name>/<run_name>_last.pth
outputs/eval_runs/<run_name>_val/summary.json
outputs/eval_runs/<run_name>_val/summary_by_count_bin.csv
outputs/eval_runs/<run_name>_val/top_abs_errors.csv
outputs/eval_runs/<run_name>_val/per_image_predictions.jsonl
outputs/eval_runs/<run_name>_test/summary.json
outputs/eval_runs/<run_name>_test/summary_by_count_bin.csv
outputs/eval_runs/<run_name>_test/top_abs_errors.csv
```

如果资源允许，再补：

```text
outputs/eval_runs/<run_name>_val/postprocess_sweep.csv
outputs/eval_runs/<run_name>_val/top_errors_by_combo.csv
```

首测验收重点：

```text
overall val MAE/RMSE
7-20 signed error
100-300 signed error
>300 signed error
avg_pre_nms / gt_avg
pred_avg / gt_avg
Top20/Top50 candidate_recall_insufficient 占比
best epoch 是否晚于旧 epoch 47
```

如果目标包含 FSCD147 detection/counting，而不只是 count MAE/RMSE，还应补充保留：

```text
outputs/eval_runs/<run_name>_val/coco_metrics.json
outputs/eval_runs/<run_name>_test/coco_metrics.json
```

并至少记录：

```text
AP / AP50 / AP75
AR_max
detection_count
image_count
```

原因是当前 checkpoint 保存仍主要受 `val_mae` / `val_rmse` 驱动；数量接近不能证明预测框与 GT 一一对齐。已有 `outputs/vis_val_3_args/coco_metrics.json` 显示 `AP=0.1078`、`AP50=0.3072`、`AP75=0.0509`，说明框定位质量和计数指标必须分开看。

## 后续项目改进方向

1. 先跑第四版 P3 首测。

   使用当前 `ScaleAwareQueryAggregator`、当前第四版配置和当前 P5 日志。不要同时启用 P3-A、P2-B、P3-B、SAM3 pooling 或 Muon。首测目标只回答一个问题：P3 是否改善高计数图有效候选召回。

2. 补 P2 verification sweep。

   先扫 `feature_similarity + soft`，再扫 `feature_similarity + sparse_hard`。不建议直接全局 hard filtering。目标是压低 `7-20` over-count，同时不让 `100-300`、`>300` signed error 更负。

3. 如果 P3 有效，再做 P3-A。

   P3-A 建议只扩展 wrapper，不改 DINOv3 源码目录：

   ```text
   DINO/FPN c1,c2,c3,c4 + class token
   -> q4 semantic anchor
   -> q4 -> q3 -> q2 -> q1 progressive query aggregation
   -> DINO exemplar similarity prior
   -> q_out dense objectness / bbox heads
   ```

   初版只启用 `c4` 语义锚点和 class-token gate，DINO consistency loss 与多层深监督放到下一轮。

4. 如果 P3/P3-A 改善召回但假阳性仍明显，再做 P2-B。

   做 count-bin prior 或 density-aware score gate。监督目标只能来自有效 `gt_bboxes` count。`density_sum_debug` 继续只用于诊断。

5. 如果候选召回基本够、少数高分假阳性拉高 RMSE，再做 P3-B。

   参考 RF-DETR mixed-query 思路，只对 NMS 前 top-K dense candidates 做轻量精修，不替换成完整 DETR decoder：

   ```text
   objectness map + bbox map
   -> top-K candidates before NMS
   -> ROI/content feature + spatial token + learned token
   -> lightweight refinement head
   -> refined score + refined box
   -> NMS
   ```

6. SAM3 只作为 exemplar mask prior。

   下一步先统计 `instances_train_sam3_1.json` 的 mask 面积、碎片度、与 exemplar box 的一致性。质量稳定后，再在 `src/datasets/data.py` 和 `src/models/DGECO.py` 加可选 mask-weighted pooling。不要把 SAM3 伪实例直接当全量 GT。

7. Muon 只做后期消融。

   等 P3/P3-A 结构固定后，再比较 AdamW 与 Muon+AdamW hybrid。第一版 Muon 只作用于新增 query/fusion/projection 隐层矩阵权重，head、bias、norm、prompt embedding 仍用 AdamW。

8. 强化学习 / policy optimization 只作为后期小规模消融。

   不建议把当前 dense detector 改成完整 RL agent，也不建议让 agent 直接在 4096 个候选框上逐个选择检测结果。动作空间太大，奖励来自 MAE/RMSE 或 NMS 后检测集合，方差高，且会把 P3 结构有效性、后处理和奖励设计混在一起。

   可以考虑两种低风险形式：

   ```text
   A. contextual bandit 后处理策略
      state: pre_nms/post_nms、score quantiles、score gap、box area stats、
             NMS reduction ratio、exemplar geometry、DINO similarity stats
      action: score_threshold / score_ratio / nms_iou / threshold_mode /
              verification_score_gamma 的离散组合
      reward: -abs(pred_count - gt_count)，或 count-bin 加权的 -MAE/-RMSE

   B. policy-gradient 式 objectness 重加权
      保留 Hungarian + focal CE + bbox L1 + GIoU 主训练
      只对 top-K 候选的 objectness loss 增加很小的全局 reward/advantage 权重
      reward 来自 NMS 后 count error 或 detection quality，不改 backbone/query 主结构
   ```

   何时可以开始改：

   ```text
   必须先满足：
   1. 第四版 P3 首测至少完成 best checkpoint 的 val/test eval。
   2. 已产出 summary.json、summary_by_count_bin.csv、top_abs_errors.csv、per_image_predictions.jsonl。
   3. 已明确主要误差来源：低计数假阳性、高计数召回不足、NMS 压制、verification 压制或阈值偏移。

   contextual bandit 可开始的条件：
   - P3/P3-A 已经让 100-300 和 >300 的 avg_pre_nms / gt_avg 上升；
   - 但 7-20 over-count 或整体 RMSE 仍被后处理参数牵制；
   - P1 后处理 sweep 和 P2 verification sweep 已经跑完，仍没有稳定默认策略。

   policy-gradient objectness 可开始的条件：
   - P3/P3-A 结构已经固定，训练曲线稳定；
   - 候选召回基本足够，但 objectness 排序与 NMS 后最终计数不一致；
   - 简单 loss reweight、threshold sweep、verification 都不能继续降低 count-bin MAE/RMSE。
   ```

   验收方式必须和普通方法同表比较：至少对比静态后处理、`regime_adaptive`、P2 verification sweep。若 contextual bandit 不能稳定优于 sweep 最优或 policy-gradient 不能改善 count-bin MAE/RMSE，应直接撤回。

9. 依赖环境保持稳定。

   `requirements.txt` 的 `torch==2.10.0+cu130` 与 PyTorch 官方 CUDA 13.0 wheel 记录匹配。第四版首测期间不要升级到 PyTorch 2.11 或改 numpy 约束；依赖升级应单独建实验。

## 暂不建议做的事

- 不要全量微调 DINOv3 backbone。
- 不要用 `density_map.sum()` 作为训练、验证或推理的 GT count。
- 不要恢复 pixel-level density loss、count loss、aux_count、density_count 作为主监督。
- 不要把 `max_detections` 大幅调低来压 MAE。
- 不要把旧 sweep 的 `combo 30` 作为默认配置。
- 不要继续只靠全局提高 `score_threshold` 或 `score_ratio` 修复误差。
- 不要第一步引入完整 RF-DETR decoder、固定 300 queries 或 DQ-DETR dynamic query selection。
- 不要把 SAM3 伪 mask 当作全部实例监督。
- 不要把 Muon 和多个结构改动放在同一次首测里。
- 不要在 P3 首测 checkpoint 评估和 P2 verification sweep 之前引入 RL。
- 不要把 dense candidate selection 做成完整 sequential RL；如需尝试，只从后处理 contextual bandit 或轻量 policy-gradient objectness 重加权开始。

## 参考资料

- DINOv3 官方介绍：https://ai.meta.com/blog/dinov3-self-supervised-vision-model/
- DINOv3 项目页：https://ai.meta.com/dinov3/
- GECO2 / Generalized-Scale Object Counting with Gradual Query Aggregation：https://arxiv.org/abs/2511.08048
- DAVE / Detect-and-Verify Paradigm for Low-Shot Counting：https://arxiv.org/abs/2404.16622
- SAM 3 / Segment Anything with Concepts：https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- RF-DETR Hugging Face 文档：https://huggingface.co/docs/transformers/main/model_doc/rf_detr
- DQ-DETR / DETR with Dynamic Query for Tiny Object Detection：https://arxiv.org/abs/2404.03507
- PyTorch previous versions：https://pytorch.org/get-started/previous-versions/
- Muon is Scalable for LLM Training：https://arxiv.org/abs/2502.16982
- Active Object Localization with Deep Reinforcement Learning：https://arxiv.org/abs/1511.06015
- Learning Globally Optimized Object Detector via Policy Gradient：https://openaccess.thecvf.com/content_cvpr_2018/papers/Rao_Learning_Globally_Optimized_CVPR_2018_paper.pdf
- Reinforcement Learning for Improving Object Detection / ObjectRL：https://arxiv.org/abs/2008.08005
- Learning non-maximum suppression：https://arxiv.org/abs/1705.02950
