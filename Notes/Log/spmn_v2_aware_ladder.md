# spmn_v2 aware ladder — 完整结论 (closed 2026-06-28)

AND adapter 之上的 "aware" 调制阶梯 (Step 0→3, codex-designed)。**已收口**:唯一有效件是
Step 1 promiscuity branch (+0.002);density-adaptive hub gate (Step 2 全局 / Step 3 pair 条件化)
实测全空。aware 定为**主表里一行小消融**,不是 contribution。headline = AND substrate + 冷启动
理论 + 胜 EmerGNN 的 fusion(见 [[spmn_v2_theory_anchor]])。

代码:`run_spmn_v2_aware.py` (--aware-step 0..3),`spmn_v2/aware_heads.py` (DecomposedStandaloneHead
+ b(v_ab) branch),`spmn_v2/aware_core.py` (AwareStructuralCore + 度数 gate global/pair),
`spmn_v2/density.py` (v_ab 6 维 + q_ab 4 维)。每步 codex review (PASS-with-fixes,已修)。

## 阶梯定义
- **Step 0** 裸 AND (参考,= 现有 standalone)。
- **Step 1** + 零初始化 promiscuity branch `s = scorer(z) + b(v_ab)`,v_ab = 6 维 per-pair 密度边际。
- **Step 2** + 全局单调 hub gate `gate = g_τ·φ_τ(log deg)`(Step 3 的对照)。
- **Step 3** + pair 条件化 `gate = g_τ·λ_τ(q_ab)·φ_τ`,q_ab = 4 维 regime(sparsity/hub_frac/...)。
- 所有 gate 零初始化 (g_τ=0 精确零起点),g_τ 投影 ≥0,clamp 上界 4,gate_reg L2+平滑。

## 结果 (test AUC, 3 seed 配对, 同 harness 同缓存)

| seed | st0 | st1 | st2 | st3 | best_ep |
|---|---|---|---|---|---|
| 42 | 0.7845 | 0.7859 | 0.7857 | 0.7857 | 1/1/1/1 |
| 43 | 0.7471 | 0.7478 | 0.7482 | 0.7480 | 1/1/1/1 |
| 44 | 0.7657 | 0.7697 | 0.7691 | 0.7692 | 1/1/1/1 |

- **Δ(1−0) branch**: mean **+0.0020** (3/3 正) — 唯一有效。
- **Δ(2−1) 全局 hub gate**: mean **−0.0001** (1/3) — 空。
- **Δ(3−2) pair 条件化**: mean **−0.0000** (1/3) — 空。
- **Δ(3−0) 全梯**: mean **+0.0019** (3/3) = 全部来自 branch。

## 为什么 hub gate 空(两个已验证原因叠加)
1. **best_ep=1 塌陷**(全程):gate 只训 ~1 epoch,g_τ 几乎没动出 0 → 学不动。
2. **结构性抵消**:全局压 hub 净≈0(诊断:瘦 pair 52% hub / 富 5%,压 hub 帮瘦伤富);pair 条件化
   在 1-epoch 内也没学出"只压瘦"。

## 塌陷根因诊断(顺带钉死,修正了之前叙事)
- `analyze_spmn_v2_struct_domain_decomp.py`:**整个 0.76 域漂移在 struct_feats 的 uncapped 计数块**
  (count_only = full = 0.76;copath 因 cp0 是空的)。
- `--zero-count-feats` retrain:**去掉计数块后塌陷照旧**(best_ep 还是 1,val 同样跳水,test −0.004)
  → **count 漂移不是塌陷成因**。
- **结论:塌陷 = 学习到的池化中介通道极快过拟合 train(普通快速过拟合),非可移除的输入协变量漂移。**
  early-stop 已抓峰值 → 报告的 0.78 没坏。codex:**别把治塌陷当主线**;walk / 支撑降级增广都被这趟
  分析判为非杠杆(避开了 build)。

## 战略close(codex)
aware 封顶 ~+0.002,塌陷是 early-stop 已处理的普通过拟合,强结果在 **AND substrate + 理论 + fusion
胜 EmerGNN**。aware = 主表一行(Step1 branch);hub gate 作 honest 负结果消融(试了 density-adaptive,
打不过简单 branch)。**转头部。**

相关:[[spmn_v2_theory_anchor]] · [[spmn_v2_adapter_asym_ab]] · [[Exps-loop-1]]
