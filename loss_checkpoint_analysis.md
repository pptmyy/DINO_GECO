# FSCD147 loss 与 checkpoint 指标分析

分析日期：2026-06-09

## 结论

当前项目的 `loss` 计算本身不是单纯的数量误差。训练时会通过 Hungarian matching 将预测框和 GT 框配对，并计算 bbox L1 loss、GIoU loss 以及 centerness/focal loss，因此它确实在监督“预测框是否和 GT 位置一致”。

真正存在风险的是 checkpoint 选择标准：`train.py` 当前保存 best checkpoint 时只看 `val_mae` / `val_rmse`，这两个指标来自“预测框个数”和“GT 个数”的差值，不检查预测框是否真的落在 GT 位置。因此，如果最终目标是检测式计数或需要可靠框位置，确实可能保存到“数量接近，但很多框位置错误”的 checkpoint。

更准确地说：

- 如果论文/实验只汇报 FSC147 count MAE/RMSE，按 `val_mae` 或 `val_rmse` 保存 count-best checkpoint 是合理的。
- 如果实验目标是 FSCD147 / detection-by-counting，即输出框也要正确，则只用 MAE/RMSE 保存 checkpoint 不充分，应该至少额外保存 `best_val_loss`、`best_val_AP50` / `best_val_AP` 或 IoU-based F1/recall checkpoint。
- `val_loss` 比 count MAE/RMSE 更贴近训练目标，但最直接对应框质量的仍然是 validation bbox AP/AP50/AP75 或固定 IoU 阈值下的 detection F1。

## 本项目代码证据

### loss 确实包含位置监督

`train.py` 的 `compute_detection_batch_loss()` 中，先从 `gt_bboxes` 构造 `target_bboxes`，再调用 `criterion(...)` 计算主分支和辅助分支 loss：

- `train.py:233-242`：构造 GT boxes，并计算 `main_losses` / `aux_losses`。
- `src/utils/losses.py:30-45`：`loss_boxes()` 对 matched predicted boxes 和 target boxes 计算 L1 bbox loss 与 GIoU loss。
- `src/utils/losses.py:48-58`：`ce_loss()` 对 centerness map 做 focal loss。
- `src/utils/losses.py:97-129`：`generate_centerness_gt()` 给 TP、FP、FN 位置构造 centerness 监督。

所以训练 loss 的方向总体是“预测框、中心响应与 GT 对齐”，不是单纯 count loss。

### checkpoint 只按数量误差保存

验证阶段统计的是预测数量与 GT 数量的差：

- `train.py:522-525`：`val_ae += abs(num_objects_gt - num_objects_pred)`，`val_rmse += pow(num_objects_gt - num_objects_pred, 2)`。
- `train.py:567-570`：得到 `current_val_mae` 和 `current_val_rmse`。
- `train.py:584-600`：只在 `current_val_rmse < best_val_rmse` 或 `current_val_mae < best_val_mae` 时保存 best checkpoint。

这里没有用 IoU、AP、matched TP 数、bbox recall 或 `val_loss` 作为 best checkpoint 条件。

### matcher 还有一个需要注意的细节

`src/models/matcher.py:79-87` 会把 IoU 最大值为 0 的 GT 标记成 non-matched，并移除对应匹配。这样做可以避免完全无交集的框被直接当作 TP，但副作用是：这些 GT 主要靠 centerness 的中心监督推动，短期内 bbox regression 不一定直接收到该 GT 的 box loss。因此，单看 count checkpoint 更不能保证框已经学好。

## 日志与已有评估证据

最新日志 `logs/log/DGECO2FSCD_20260608_163211/metrics.jsonl` 显示 loss 和 count metric 可以明显不同步：

- epoch 4：`val_loss=1.4544`，`val_mae=50.2776`，`val_rmse=89.7086`，会被 count-best 逻辑保存。
- epoch 13：`val_loss=1.2554`，`val_mae=58.2636`，`val_rmse=92.9997`，loss 更低，但 count metric 更差，因此不会成为 best count checkpoint。

另一个较长 run `logs/log/DGECO2FSCD_20260602_101741/metrics.jsonl`：

- best loss：epoch 57，`val_loss=0.8286`，`val_mae=20.8476`，`val_rmse=51.0233`。
- best MAE/RMSE：epoch 47，`val_loss=0.8444`，`val_mae=20.7030`，`val_rmse=49.5592`。

这说明 count-best 和 loss-best 不总是同一 epoch。

已有 bbox 评估文件 `outputs/vis_val_3_args/coco_metrics.json` 中：

- `AP = 0.1078`
- `AP50 = 0.3072`
- `AP75 = 0.0509`
- `image_count = 1286`
- `detection_count = 73171`

这进一步支持你的担心：数量指标和 bbox 定位质量不是同一个问题，count 接近并不等价于框位置正确。

## 近期 FSCD147 / FSC147 相关论文代码如何处理

### Counting-DETR / FSCD147

Counting-DETR 提出 FSCD-147，原因是原始 FSC147 只有所有目标的 dot annotation 和 3 个 exemplar boxes，并没有所有目标的完整 bbox；FSCD-147 为 val/test 补充了所有目标框，用于 few-shot counting and detection 评估。

公开论文和 repo 的关键做法：

- 任务定义本身要求同时输出 count 和 object bounding boxes。
- 训练使用 DETR 风格 Hungarian matching，并监督 classification + bbox。
- 论文明确报告 counting 和 detection 两类指标，而不是只用 count 证明检测正确。
- 代码中 checkpoint 主要是周期保存；推理会导出 COCO-style `predictions.json`，用于后续 bbox 评估。

本地核对到的代码点：

- `CountDETR_147_2nd_stage/models/anchor_detr.py:143-148`：SetCriterion 注释说明先做 Hungarian assignment，再监督 matched GT/prediction 的 class 和 box。
- `CountDETR_147_2nd_stage/models/anchor_detr.py:213-233`：bbox L1 和 GIoU loss。
- `CountDETR_147_2nd_stage/models/matcher.py:235-242`：matching cost 包含 bbox L1、class、GIoU。
- `CountDETR_147_2nd_stage/main.py:222-235`：checkpoint 周期保存，不是只根据 count-best 保存。
- `CountDETR_147_2nd_stage/engine.py:155-172`：推理导出带 `bbox`、`score` 的 `predictions.json`。

### GeCo / NeurIPS 2024

GeCo 与当前项目最接近：也是 detection/segmentation/counting 统一式低样本计数。其公开说明强调，传统 surrogate center loss 不直接优化检测任务，可能导致 suboptimal counts；GeCo 目标是直接优化 detection/counting。

公开 repo 的处理方式是把 count evaluation 和 bbox evaluation 分开：

- `evaluate.py`：后处理预测框，统计 MAE/RMSE，并导出预测 bbox。
- `evaluate_bboxes.py`：单独跑 COCO bbox evaluation，报告 `AP`、`AP50`、`AP75`、`APs`、`APm`、`APl`。
- `utils/losses.py`：训练 loss 同样包含 bbox L1/GIoU 与 centerness/center supervision。

本地核对到的代码点：

- `GeCo/utils/losses.py:282-300`：bbox L1 + GIoU。
- `GeCo/utils/losses.py:335-367`：TP/FP/FN 的 centerness target。
- `GeCo/evaluate_bboxes.py:241-244`：输出 count MAE/RMSE。
- `GeCo/evaluate_bboxes.py:271-314`：COCO bbox evaluation，并报告 `AP`、`AP50`、`AP75` 等。

### PseCo / CVPR 2024

PseCo 是 detection-based counting 框架，使用 SAM proposals 和 CLIP 分类。它也没有把 count metric 当作框定位正确的证明，而是：

- 评估时先用 Detectron2 `COCOEvaluator(..., tasks=['bbox'])` 得到 bbox detection metrics。
- 同一个 evaluate 函数里再单独根据 score threshold 统计 count MAE/RMSE/NAE/SRE。
- validation 上扫描 threshold 以最小 MAE 选择 counting threshold，然后用于 test count。
- 训练 proposal classifier 时，用 `IoU > 0.5` 将候选框标为正样本，直接把框和 GT 的重叠关系纳入训练标签。

本地核对到的代码点：

- `PseCo/fsc147/4_1_train_roi_head.py:97-164`：`COCOEvaluator(tasks=['bbox'])` 计算 detection bbox metrics。
- `PseCo/fsc147/4_1_train_roi_head.py:166-203`：单独计算 count MAE/RMSE/NAE/SRE。
- `PseCo/fsc147/4_1_train_roi_head.py:185-196`：validation 扫描 threshold，以 MAE 最小选择计数阈值。
- `PseCo/fsc147/4_1_train_roi_head.py:234-244`：用 `box_iou` 和 `IoU > 0.5` 标正样本。
- `PseCo/fsc147/4_1_train_roi_head.py:301-304`：先在 val 定 threshold，再评 test，并周期保存 checkpoint。

## 对当前项目的判断

你的判断是成立的：如果保存 checkpoint 的目标是“框预测与 GT 保持一致”，当前只按 `val_mae` / `val_rmse` 保存并不充分。MAE/RMSE 只能说明数量误差小，不能排除以下情况：

- 预测数量正确，但框整体偏移。
- 一部分 GT 漏检，另一部分位置错误的 FP 数量刚好补上。
- NMS/verification 后数量接近，但 matched IoU 很低。
- 框尺寸错误，AP75 很差，但 count 仍然接近。

从 FSCD147 相关论文代码看，更常见也更严谨的处理方式是：

- count 指标：MAE/RMSE，用来选择或报告 count-best。
- detection 指标：AP/AP50/AP75、IoU-based precision/recall/F1，用来选择或报告 detection-best。
- training loss：bbox L1/GIoU + classification/centerness/matching loss，用来优化位置。
- threshold：可以用 validation MAE 调整 count threshold，但不能把 threshold-best MAE 当作 bbox-best。

## 建议的 checkpoint 策略

不建议简单把现有 `best_val_mae` / `best_val_rmse` 删除或替换。更稳妥的是保留多种 best：

- `best_val_mae.pth`：用于 count-only 结果。
- `best_val_rmse.pth`：用于 count-only 结果。
- `best_val_loss.pth`：更贴近训练目标，包含 bbox/centerness。
- `best_val_ap50.pth` 或 `best_val_ap.pth`：用于 detection/localization 结果，最能回答“框是否在 GT 位置”。

如果算 AP 成本太高，可以先用轻量替代指标：

- 对每张图做预测框与 GT 框 IoU matching。
- 统计 `TP@IoU=0.5`、`FP`、`FN`。
- 保存 validation F1 或 recall 最好的 checkpoint。

## 参考资料

- Counting-DETR paper: https://arxiv.org/abs/2207.10988
- Counting-DETR repo: https://github.com/VinAIResearch/Counting-DETR
- GeCo paper: https://arxiv.org/abs/2409.18686
- GeCo repo: https://github.com/jerpelhan/GeCo
- PseCo repo: https://github.com/Hzzone/PseCo
- COCO evaluation implementation: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

## 本次操作说明

按照要求，本次没有修改训练代码。仅创建了本文档，并在分析完成后删除用于只读调研的临时 clone 目录。
