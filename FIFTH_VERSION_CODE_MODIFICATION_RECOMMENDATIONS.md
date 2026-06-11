# 第五版代码修改方案：分阶段训练与双指标门控

更新时间：2026-06-09

本文从第四版文档中的“分阶段训练与双指标门控计划”独立出来，作为第五版方案。目标不是替换当前 detection-only 主线，而是在 P3/P3-A 等结构验证后，把“框定位正确”和“最终计数准确”拆成可控阶段，避免只用 MAE/RMSE 保存到数量接近但位置错误的 checkpoint。

## 方案目标

第五版围绕两个指标体系推进：

```text
检测质量：AP / AP50 / AP75 / Recall@IoU=0.5 / F1@IoU=0.5
计数质量：MAE / RMSE / count-bin signed error
```

最终模型选择必须说明 checkpoint 来源：

```text
detection-best checkpoint: 用于报告 bbox 定位质量
count-best checkpoint: 用于报告计数质量
detection-gated count-best checkpoint: 在检测质量不明显退化的前提下选择计数最优
```

不能再默认用 `best_val_mae` 或 `best_val_rmse` 代表 bbox-best。

## 阶段 0：现有 checkpoint 的检测/计数复核

当前即可执行，不需要改训练代码。对现有候选 checkpoint 统一运行 `tools/eval_checkpoint.py --coco-eval`，至少比较：

```text
*_best_val_mae.pth
*_best_val_rmse.pth
*_last.pth
```

输出同时看：

```text
summary.json: MAE / RMSE / count-bin signed error
coco_metrics.json: AP / AP50 / AP75 / AR_max
per_image_predictions.jsonl: 高误差样本和错误框分布
```

判断方式：

```text
如果 best_val_rmse 的 AP50/AP75 明显低于 last 或其他 checkpoint，
说明“count-best 不等于 bbox-best”已经在当前项目中实际发生。
```

## 阶段 1：检测/定位优先训练

前提：P3 首测完成，且 bbox-aware eval 口径已固定。

训练目标仍保留 Hungarian matching、focal CE、bbox L1、GIoU 和 centerness/objectness 监督，不建议先改 loss 主体。关键改动是增加 validation detection metric，并额外保存：

```text
*_best_val_ap50.pth
*_best_val_ap.pth
*_best_val_f1_iou50.pth  # 如果 COCO AP 每轮成本太高，可先用轻量 IoU-F1
*_best_val_loss.pth
```

该阶段主看：

```text
AP50 / AP75
Recall@IoU=0.5
F1@IoU=0.5
val_loss
```

不应只用 `val_mae` / `val_rmse` 选择检测模型。`best_val_mae` 和 `best_val_rmse` 可以继续保存，但只代表 count-best。

## 阶段 2：计数校准

前提：阶段 1 已得到 bbox 质量可接受的 detection-best checkpoint，且候选召回不是主要瓶颈。

优先不训练新结构，先做 validation 后处理搜索：

```text
score_threshold
score_ratio
threshold_mode
nms_iou
verification_mode / verification_threshold / verification_score_gamma
max_detections
```

选择规则不要只看 MAE/RMSE，使用检测门控：

```text
先筛选：AP50 >= best_AP50 * 0.98，或 F1@IoU=0.5 下降不超过 2%
再选择：val MAE / RMSE 最低的后处理组合
```

如果只靠阈值搜索无法稳定改善，可以冻结 backbone 和 bbox regression，仅微调 objectness / verification / score calibration 分支。

## 阶段 3：可选联合微调

前提：阶段 2 后 count 改善但 AP/F1 有下降，或者 detection-best 与 count-best 差距过大。

做小学习率联合微调，同时保留多类 best checkpoint：

```text
best_val_ap50
best_val_f1_iou50
best_val_mae
best_val_rmse
best_val_loss
```

联合微调不能只按 count metric 早停。每次 count 变好，都必须同步检查 AP50/AP75/F1 是否下降。

## 阶段 4：强化学习 / policy optimization

RL 不适合作为当前训练主线，也不适合替代 bbox/GIoU 监督。它只适合作为后期后处理策略学习或轻量 objectness 重加权。

低风险形式：

```text
A. contextual bandit 后处理策略
   state: pre_nms/post_nms、score quantiles、score gap、box area stats、
          NMS reduction ratio、exemplar geometry、DINO similarity stats
   action: score_threshold / score_ratio / nms_iou / threshold_mode /
           verification_score_gamma 的离散组合
   reward: F1@IoU=0.5 - count_error_penalty - FP_penalty

B. policy-gradient 式 objectness 重加权
   保留 Hungarian + focal CE + bbox L1 + GIoU 主训练
   只给 top-K 候选 objectness loss 增加很小的 reward/advantage 权重
   reward 来自 NMS 后 count error 和 detection quality，不改 backbone/query 主结构
```

不建议做：

```text
把 dense detector 改成完整 sequential RL agent
让 agent 在 4096 个候选框上逐个选择保留/删除
用纯 MAE/RMSE reward 训练 detector 主干
在 P3 首测前引入 RL
```

## 执行顺序

结合当前第四版 P3/P2 状态，执行顺序应为：

```text
现在：
  只做阶段 0。复核 best_val_mae / best_val_rmse / last 的 AP/AP50/AP75。
  不启动分阶段重训，不引入 RL。

第四版 P3 首测完成后：
  如果 P3 没有改善高计数候选召回，先处理 P3/P3-A 或候选生成问题；
  如果 P3 改善候选召回，但 AP50/AP75 与 count-best 明显不一致，
  开始阶段 1，加入 bbox-aware checkpoint。

P2 verification sweep 和后处理 sweep 完成后：
  如果 detection-best 已稳定，但 MAE/RMSE 仍受阈值、NMS 或 verification 牵制，
  开始阶段 2 的计数校准。

结构稳定且普通方法到达瓶颈后：
  如果静态后处理、regime_adaptive、P2 verification sweep 都无法继续改善，
  再把阶段 4 的 contextual bandit 作为小规模消融。

不建议在当前 P3 首测前施行：
  多阶段重训、policy-gradient objectness、完整 RL、或任何会改变主训练闭环的策略。
```

## 验收原则

```text
检测表使用 detection-best checkpoint：AP / AP50 / AP75 / F1@IoU=0.5
计数表使用 count-best 或 detection-gated count-best：MAE / RMSE
最终模型选择必须说明 checkpoint 来源，不能用 count-best 直接代表 bbox-best。
```

第五版能否进入代码实现，取决于阶段 0 和第四版 P3 首测后的证据。若现有 checkpoint 的 detection-best 与 count-best 差异不明显，则优先继续完成 P3/P2 sweep；若差异明显，再把 bbox-aware checkpoint 保存和 detection-gated count selection 纳入训练主流程。
