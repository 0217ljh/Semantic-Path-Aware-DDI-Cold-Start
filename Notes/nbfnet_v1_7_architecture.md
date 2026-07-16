# NBFNet v1.7 架构示意 (cold-start S2 binary DDI)

vanilla NBFNet (Zhu et al., NeurIPS 2021) 在 merged KG 上的 forward 架构 + 训练冷启动机制。
代码对应 `Code/my_code/models/nbfnet_v1_7/{nbfnet_model.py, nbfnet_trainer.py}`。
v1.71 = 同架构 + 运行时加速 (TF32/AMP/eval 调度)，见文末。

---

## 总架构 (一个 BF tower 展开, 另一侧对称)

```
   Input pair (a,b)
        │
        ▼
   ── 冷启动模拟 (shuffle_train S2) ───────────────────────────────
   每 epoch: 随机 ~20% 药 = emerging。 target=(a,b) 两端都 emerging
   ⇒ epoch KG 里 a、b 的所有 DDI 边全部不在  (= 真·冷启动, 按药物全局删)
   facts = 仅 kept-kept 的 DDI  +  base KG (非 DDI)
   augment 逆边;  query-edge mask 删 (a,r_ddi,b) 及逆 = S2 下 no-op
        │
        ▼
 ╔════════════════════════════════════════════════════════════════╗
 ║  BF pass : source = a    (source=b 对称跑一遍, 此处略)          ║
 ║                                                                  ║
 ║  h0:  h0[a] = q  ,  其它节点 = 0     (INDICATOR 点火)            ║
 ║   │                                                              ║
 ║   ▼  ┌─ 第 1 层 ────────────────────────────────────────────┐  ║
 ║   ●  │ w_q(r) = W_r^(1)·q + b_r^(1)      ← 本层独占 (n_rel,d,d)│  ║
 ║      │ msg = h_x ⊙ w_q(r)  (DistMult)                          │  ║
 ║      │ ⊕ 重注入 h0 ; PNA(mean/max/min/std × id/amp/atten); ReLU│  ║
 ║   │  └──────────────────────────────────────────────────────┘  ║
 ║   ▼  ┌─ 第 2 层 ────────────────────────────────────────────┐  ║
 ║   ●  │ w_q(r) = W_r^(2)·q + b_r^(2)      ← W_r^(2) ≠ W_r^(1) │  ║
 ║      │ msg/⊕h0/PNA/ReLU  (同结构, 不同权重)                   │  ║
 ║   │  └──────────────────────────────────────────────────────┘  ║
 ║   ▼   ⋮   (每层一套独立 W_r^(t), b_r^(t) — 不变式4)             ║
 ║   ▼  ┌─ 第 L 层 ────────────────────────────────────────────┐  ║
 ║   ●  │ w_q(r) = W_r^(L)·q + b_r^(L)                          │  ║
 ║      │ msg/⊕h0/PNA/ReLU                                       │  ║
 ║   │  └──────────────────────────────────────────────────────┘  ║
 ║   ▼  读出 b:  hf = h_a^(L)[b]   (d,)                            ║
 ╚════════════════════════════════════════════════════════════════╝
        │  (另一塔: source=b → hr = h_b^(L)[a])
        ▼
   h_sym = hf + hr            (表示级对称, 不变式7)
        ▼
   logit = MLP( concat[ h_sym , q ] )    (不变式8)
        ▼
   → DDI 二元分类 score
```

---

## W_r 为什么每层不同 (不变式4)

```
              层 t=1        层 t=2        ...   层 t=L
            ┌─────────┐  ┌─────────┐         ┌─────────┐
  W_r:      │(n_rel,  │  │(n_rel,  │   ...   │(n_rel,  │   ← 每层一个独立张量
            │  d, d)  │  │  d, d)  │         │  d, d)  │     (NBFLayer 各自持有)
            └─────────┘  └─────────┘         └─────────┘
   每层内部再按关系切片:  W_r[0], W_r[1], ..., W_r[n_rel-1]  ← 每个关系一个 d×d 矩阵
```

- `NBFLayer.__init__`: `self.W_r = nn.Parameter(torch.empty(n_rel, d, d))`，每层独占。
- `NBFNetDDI`: `self.layers = ModuleList([NBFLayer(...) for _ in range(L)])` → L 套独立 W_r。
- 第 t 层算 `w_q(r) = W_r^(t)·q + b_r^(t)`（`compute_w_q`, einsum `"rij,j->ri"`）。
- 意义: 同一种关系 r 出现在第 1/2/3 跳会被**不同矩阵**变换 ⇒ 模型能区分关系在路径上的
  顺序/位置 (noncommutative path)。若全层共享一个 W_r ⇒ 退化成位置无关。
- v1.71 merged 实例: 每层 `W_r=(38,16,16)`（n_rel=2×19），共 L=3 层。

---

## 边上传的是什么 + 波前怎么外扩

```
        节点 x (当前状态 h_x^(t-1), d 维)
            │   沿关系 r 的边 x ──r──► v
            ▼
   msg(x→v) = h_x^(t-1)  ⊙  w_q(r)        ← 逐元素乘 (DistMult)
                          └ w_q(r) = W_r^(t)·q + b_r^(t)   (这一层、这个关系的变换)
            │
            ▼
   v 收到所有入边 msg + 自己的 h0  ──PNA聚合──► h_v^(t)
```

波前 (source=a, 小例子 a─r1─►m─r2─►b)：

```
   节点:        a            m            b          其它
 ─────────────────────────────────────────────────────────
 t=0 (点火)   h_a=q         0            0           0          只有 a 亮
                │ 送 q⊙w_q(r1)
                ▼
 t=1         h_a(重注入q)  h_m=PNA(q⊙w_q(r1))  0       0        m 被点亮(1跳)
                              │ 送 h_m⊙w_q(r2)
                              ▼
 t=2         h_a           h_m          h_b=PNA(h_m⊙w_q(r2))   0  b 被点亮(2跳)
 ...
 t=L         所有 ≤L 跳可达节点都已点亮; 读出 h_a^(L)[b] = h_q(a,b)
             = a→…→b 所有 ≤L 跳路径证据的聚合
```

- h 是整张图状态矩阵 `(n_nodes, d)`；一次 BF 同时更新全图，最后只读出 b 那一行。
- 每层都重注入 h0 (不变式3)：source a 每层被按住在 q，火源不熄；其它节点 h0=0。
- 没被 a 在 L 跳内触达的节点状态恒 0 → 对 b 的读出无贡献（这就是路径推理）。

---

## 两层防泄漏 (作用层级不同)

| 机制 | 层级 | 作用 | S2 下 |
|---|---|---|---|
| `shuffle_train(S2)` | epoch-KG (药物级) | target=(a,b) 两端 emerging ⇒ a、b 的**所有** DDI 边都不在 epoch KG | 这才是"删 a、b 全部 DDI"的地方 |
| query-edge mask | per-pair (关系感知) | 删直接的 (a, r_ddi, b) 及逆边 (NBFNet 标准目标边掩码) | **no-op**（a、b 本就无 DDI 边）|

结论: 冷启动"删 a、b 全部 DDI"由 shuffle_train 负责（按药物全局删），不是 query-mask。
query-mask 在 S2 是冗余 no-op（`_query_edges_present` 返回 False → 走批量无掩码路径）。
eval 端: test 的 a、b 是 G2，从不出现在 train_ddi → eval KG 里本就无其 DDI 边，一致。

---

## 8 条不变式落点

| # | 不变式 | 位置 |
|---|---|---|
| 1 | score head 无 drug embedding | `mlp_head(concat[h_sym, q])` |
| 2 | drug_a = INDICATOR 锚点 h⁰=q | `bellman_ford` h0 赋值 |
| 3 | 每层边界重注入 | NBFLayer 的 ⊕ h0 |
| 4 | per-(层,关系) 变换 | 每层独立 W_r^(t), b_r^(t) |
| 5 | 逆边增广 (非自环) | `augment_inverse_edges` |
| 6 | 关系感知 query-edge mask | `build_union_query_edge_mask` (S2 no-op, 经 `_query_edges_present`) |
| 7 | 表示级对称 | `h_sym = hf + hr` |
| 8 | MLP 输入 concat([h_sym,q]) | 打分头 |

---

## v1.71 相对此架构的差异 (仅 3 处加法, 不改 BF 数学)

- `init_model`: 开 TF32
- `_score_pairs`: 外包 `torch.autocast(bf16)`，输出 logits 升回 fp32
- `fit`: eval 改成每 N epoch + 打 train/eval 计时拆分

实测 (merged seed42): 全程 8.35h → 0.56h (~15×), test_s2 AUC 0.7911 → 0.7943 (parity)。
```
