# EmerGNN 传播加速:torchdrug generalized_rspmm → CSR sparse.mm 迁移

2026-07-01。codex-reviewed (thread 019f1f32)。目标:torchdrug 的核心加速(融合
`generalized_rspmm` CUDA kernel)能否迁移到我们纯 PyTorch EmerGNN,好在 5090 上跑
更大/更公平的 KG。结论:**能,且已 ship**——不用 torchdrug 包,不写 CUDA/Triton。

## 核心洞察
`generalized_rspmm` 语义 = 每条边 (h,t,r): `out[t] += rel_t[r]*hiddens[h]`。因为
`rel_t[r,b,:]` 对同一关系 r 的所有边是**常量**(不依赖端点),求和可因式分解:

    out[t,b] = Σ_r rel_t[r,b] ⊙ (A_r @ H)[t,b]        A_r[t,h]=关系 r 的 (h→t) 边计数

→ 重写成**逐关系稀疏矩阵乘**,永不物化 (E,B,D),复杂度 N·B·D 而非 E·B·D,纯 autograd
(无需手写 backward),自动兼容 sm_120。

## 环境事实(实测)
torch 2.7.1+cu128 / CUDA 12.8 / RTX 5090 sm_120;torch_scatter 2.1.2+pt27cu128 可用;
triton 3.3.1;nvcc 12.8;**torchdrug 包装不上(CUDA 不兼容,符合预期)**。

## 三条路 + 选型(codex)
- A 抽 torchdrug .cu 现编 → 不值得(构建/API 漂移风险);
- B Triton 重写融合 kernel → 后手,CSR 不够再上;
- **C 纯 torch sparse.mm → 选它**。但 **COO 反而慢**(PyTorch COO spmm kernel 差),
  **必须 CSR**(cuSPARSE 后端)。

## Benchmark(传播 kernel,L=3,N=21k E=307k B=32 D=64 all_rel=11,fwd+bwd,5090)
| 路径 | ms/iter | 峰值显存 |
|---|---|---|
| (A) chunk+checkpoint(现状省显存版) | 157.7 | 4974 MB |
| (A') chunk 无 checkpoint | 121.2 | **22484 MB**(大 KG 会爆) |
| (B) COO sparse.mm | 178.3 | 6897 MB |
| **(C) CSR sparse.mm** | **93.2** | 6859 MB |
→ **C 比 A 快 1.69x**,比 A' 快 1.30x 且省 3.28x 显存。

## 数值等价(全 PASS)
- fp64 gradcheck PASS;fp64 精确等价 fwd+bwd 到 ~1e-14~1e-16(含 N=21k);
- fp32 CUDA 相对误差 ≤1.6e-5 = 纯求和顺序舍入(同规模 fp64 精确到 3.9e-16 为证);
- 边界覆盖:duplicate edges / self-loop / 空关系槽 / 偏斜计数;CSR ≡ COO ≡ edge-list;
- **全模型 parity**:同权重 EmerGNN vs EmerGNNFast,logits rel 2.4e-7,最差梯度 rel 2.7e-7。

## 交付物(新文件,未改任何现有文件)
- `Code/baseline/emergnn/rspmm_sparsemm.py` — CSR/COO 稀疏乘 kernel + 边列表参考实现。
- `Code/baseline/emergnn/model_fast.py` — `EmerGNNFast(EmerGNN)`,只 override `_propagate`,
  指纹式 adjacency cache(顺序无关 → shuffle_train 重排无害;device/dtype/content 变则失效)。
- `Code/scripts/test_rspmm_equivalence.py` / `bench_rspmm_speed.py` / `test_emergnn_fast_parity.py`。

## 待办(需用户拍板 / 后续)
1. **接入注册 baseline**:`_per_mode.py::fit()` line 366 + `load()` line 714 内联
   `EmerGNN(...)`,无 `_make_model` 钩子。要注册 `emergnn_fast` 需动 `_per_mode.py`
   (受"改现有文件先确认"约束)→ **待用户确认**再做。方案:加一个 `_make_model` 钩子 +
   `_PerModeEmerGNN` 子类返回 `EmerGNNFast`,不破坏现有类。
2. **验收 gate**:3-seed 同配置 emergnn vs emergnn_fast,AUROC 统计一致(仅 fp32 舍入差)。
3. **大 KG 扩展**(codex:measure-then-optimize):HetioNet 级(~3.66M 边、~109 关系)
   → 每层 109 次 sparse.mm。先 profile;若 launch 开销成瓶颈,先做关系分块合并,再考虑
   CUDA graphs / per-layer checkpoint,**别提前优化**。
4. 公平对比:在更大/更富 KG 上重跑 EmerGNN,分离"架构优势 vs KG-access 优势"。

相关:[[spmn_v2_theory_anchor]] · [[project_semantic_path_ddi]] · [[feedback_scripts_persist_logs]]
