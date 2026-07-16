# spmn_v2 adapter — 非对称分层 A/B (SUM support)

2026-06-26. 第一块砖:在 SUM(L=5) 支撑集上,把 within-τ 池化切成 sym(d_a=d_b)/asym(d_a≠d_b)
双通道,对比不切。seed42,d=32,80 epoch,k_per_type=128,n_max=1200,copath OFF。

## 结果 (test_s2)

| 变体 | val AUC | val AUPRC | test AUC | test AUPRC |
|---|---|---|---|---|
| SUM asym OFF | 0.7471 | 0.7566 | 0.7352 | 0.7500 |
| SUM asym ON  | 0.7501 | 0.7559 | **0.7491** | **0.7608** |
| Δ (on−off)   | +0.0030 | −0.0007 | **+0.0139** | **+0.0109** |

参考:AND no-rel 0.7720 / rel-head(AND, copath ON) 0.7794 / NBFNet merged 0.7943。
结果文件:`Code/runs/spmn_v2_standalone/sum_asym{0,1}_absdiff0_seed42_lmax5_d32.json`。

## 结论

- **非对称机制有效**:asym ON > OFF,test +1.4pt AUC / +1.1pt AUPRC。必须显式分层才提得出
  SUM 引入的非对称介导的信号。这是 adapter 立论的第一个正向验证。
- **但 standalone 水位低**:SUM asym-on 0.7491 仍低于 AND rel-head 0.7794、NBFNet 0.7943。
  说明 **SUM 支撑集 standalone 比 AND 还差** —— n_max=1200 灌进的远-远 hub 稀释了信号,
  非对称分层只捞回一部分,没翻盘。
- **单 seed,val/test 不一致**(test +1.4 但 val 只 +0.3,val AUPRC 微降)→ 增益需多 seed 确认。
- copath OFF,和 0.7794 非同条件(0.7794 带 copath),别直接对。

## 方法学坑(已记)

- `compute_struct_features(with_copath=True)` 对支撑集每个中介做一次 BFS;SUM ~1000 中介/对 →
  不可行(几百小时)。SUM 实验一律 copath OFF(与 asym 正交)。详见 run_spmn_v2_standalone.py。
- build_pair_support 的 cap 按 −(d_a+d_b) 排序,会系统性淘汰高 sum 的非对称介导 → SUM 需调大
  k_per_type/n_max(用了 128/1200)否则非对称还没进模型就被裁。

## 下一步

1. **走 A:E0 互补性闸门**(read-only)。NBFNet 已存 per-pair 预测
   (`...nbfnet_v1_71_merged_seed42.../test_s2_scores.npz`:pair_a/pair_b/y_true/y_score)。
   给 standalone 加了 `--save-predictions`。脚本:`analyze_spmn_v2_complementarity.py`
   (rank-norm 凸组合 AUC 扫描 + 错误集重合)。凸组合若超过纯 NBFNet 0.7943 → 融合有 headroom。
2. 若互补 → 建 frozen-NBFNet + FusionHead(codex 方案,先冻结避免 train 泄漏)。

## 后续目标(明确记下)

**去除 SUM 里的噪声影响。** standalone SUM < AND 的根因是远-远 hub 稀释。候选方向:
- cap 策略改进(当前 −(d_a+d_b) 排序对非对称不公平;设计一个保非对称、压远-远对称 hub 的打分);
- n_max 扫描(1200 太松,试 400/600);
- 在 asym 通道内进一步按 |d_a−d_b| / 距离分档(absdiff embed,已留 flag);
- degree-aware 加权(AA)在池化里更激进地压 hub。
目标:让"非对称介导"的信号在更干净的支撑集上更突出,缩小与 AND/NBFNet 的差距。

## 2026-06-26 续:dist-attn + 纯结构 ablate + 塌陷诊断

**dist-attn(距离条件化注意力,零初始化 (d_a,d_b) 偏置表)2×2(test AUC):**

| | dist OFF | dist ON |
|---|---|---|
| asym OFF | 0.7352 | 0.7454 |
| asym ON | 0.7491 | 0.7502 |

→ dist-attn 和 asym 高度冗余(dist 救 no-split +1.0pt,叠在 split 上仅 +0.1pt)。感受野去噪到顶 ~0.75,远低于 AND 0.779 / NBFNet 0.794。

**纯结构 ablate(`--no-entity-embed`,asym ON):** test 0.7511 / val 0.7459,params 5.71M → **8.7K(−650×)**,test 不降反微升。
→ **entity_embed 是死重**(无净可迁移信号),模型可做成极简纯归纳。

**塌陷诊断(决定性):** 纯结构(8.7K 参数)的 val 曲线 **仍从 ep3 的 0.746 塌到 ep80 的 0.670**(−7.6pt),train loss 单调降。
→ **根因②(entity 记忆)被证伪**(删光照塌);**根因①(训练分布失配)坐实**。即便特征可迁移,学到的"结构→标签"映射按 seen-drug 拟合,迁移不到 unseen。容量无关。

**结论:唯一根上解 = 冷启动模拟训练(根因①)。** NBFNet 用 shuffle_train,val 曲线在 0.745–0.775 平台不塌(best ep8);我们的 adapter 不模拟冷启动 → 塌。下一步:给 adapter 训练加 drug 级 holdout 的冷启动模拟。

**工程:** standalone 现在落盘 `val_curve` + `best_epoch`(见 [[feedback_scripts_persist_logs]])。

## 2026-06-26 再续:B(冷启动模拟)证伪 + 协变量漂移诊断

**B(NBFNet 式 shuffle_train)对 adapter 是 no-op(codex 代码核实):** 支撑集建在 DDI-masked KG([retrieval.py:25](Code/my_code/models/spmn_v1/retrieval.py:25))且排除 drug 中介([retrieval.py:376](Code/my_code/models/spmn_v1/retrieval.py:376))→ 没有 DDI 捷径可删;adapter 无 drug-specific 参数 → episodic drug-holdout 无容量可保护。我们等于已隐式处于 NBFNet+shuffle_train 状态。

**塌陷真因 = 协变量漂移(诊断坐实,`analyze_spmn_v2_domain_shift.py`):**
- 域分类器 train vs val_s2 GBDT **AUC 0.760**(linear 0.605),train vs test_s2 **0.775** —— 远超 0.5。
- 按 label:正 0.749 / 负 0.707(两类都漂,正略多)。
- 方向:unseen pair 的**未裁剪 per-type 中介计数更少**(protein 3490→3323 −4.8%,BP 632→597,pathway 134→119…),即 **unseen drug KG 注释更稀疏**。capped 总支撑集 854→865 几乎不变(cap 拉平)。
- 结论:模型拟合 train 富计数 regime,迁移不到 test 稀计数 regime → 塌。**容量无关、B 无关,是域偏移。**

**正确的杠杆(若推 standalone)= 域泛化**,非 DDI-masking:drug-disjoint 伪-S2 episodes + CORAL/MMD/对抗域分类器/Group DRO on `z_adapter`。这也是一个 paper-worthy 分析点。
**但塌陷对融合 moot**(best_state 取峰值,E0 已证融合 0.805)→ 融合(C)仍是最快 payoff。

相关:[[project_semantic_path_ddi]] · [[project_ddi_notes_log_convention]]
