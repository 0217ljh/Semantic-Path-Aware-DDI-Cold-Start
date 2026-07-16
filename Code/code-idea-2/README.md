# code-idea-2

Idea2 专属代码目录，与正在运行的另一条 idea **物理隔离**，互不干扰。

设计文档：`Notes/Ideas/idea2/idea2-design.md`
决策流水：`Notes/Log/idea2/idea2-decision-log.md`

## 约定

- 本目录只放 idea2 的代码。不改动 `Code/scripts/`、`Code/my_code/` 等已有逻辑（另一条 idea 在用）。
- 只读访问 `Code/data/`（KG、DDI、splits）。不写不改。
- WSL conda env `project_1` 运行；cwd = 项目根目录。
- 实验产物写到 `Code/runs/`（新 run_id，不覆盖历史）。

## 当前阶段

Stage 1（GATE，未实现）：label-blind 链补全 → 断链恢复率 + 初步预测准确率探针。
详见设计文档 §6。

## 待 KG agent 完成后再动

KG 分层骨架 / 实体对齐 / 关系类型粒度由另一 agent 处理中，完成后再对接（见设计文档 §9 pending）。
