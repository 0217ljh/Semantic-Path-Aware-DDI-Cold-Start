---
title: Semantic-Path-Aware-DDI-Cold-Start
project: Semantic-Path-Aware-DDI-Cold-Start
type: project-readme
status: active
tldr: Semantic path guided cold-start ddi prediction
start_date: 2026-05-11
deadline: 
tags: ["#project", "#project/Semantic-Path-Aware-DDI-Cold-Start"]
created: 2026-05-11
---

# Semantic-Path-Aware-DDI-Cold-Start

> Semantic path guided cold-start ddi prediction

## At a Glance

```dataviewjs
const base = `03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Notes`;
const insights = dv.pages(`"${base}/Settings/Insights"`).where(p => p.type === "proj-insight").length;
const settingExists = dv.page(`${base}/Settings/Setting.md`) ? "✓ locked" : "—";
const papers = dv.pages(`"${base}/Settings/Paper-Review"`).where(p => p.type === "proj-paper-note").length;
const ideas = dv.pages(`"${base}/Ideas"`).where(p => p.type === "proj-idea");
const running = ideas.where(i => i.status === "running" || i.status === "open").length;
const won = ideas.where(i => i.status === "won").length;
const lost = ideas.where(i => i.status === "lost" || i.status === "abandoned").length;
const exps = dv.pages(`"${base}/Experiments"`).where(p => p.type === "proj-explog").length;

dv.table(
  ["Stage", "Status"],
  [
    ["1 — Insights", `${insights} note(s)`],
    ["2 — Setting", settingExists],
    ["3 — Paper Review", `${papers} paper(s)`],
    ["5 — Ideas", `${running} running · ${won} won · ${lost} dropped`],
    ["5.1 — Experiments", `${exps} log(s)`]
  ]
);
```

---

> [!success]+ Stage 5 — Current Ideas
> The primary progress view. Each idea has its iteration log + linked experiments.
>
> ```dataview
> TABLE WITHOUT ID
>   file.link AS "Idea",
>   status AS "Status",
>   tldr AS "Hypothesis"
> FROM "03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Notes/Ideas"
> WHERE type = "proj-idea"
> SORT status ASC, created DESC
> ```

> [!info]+ Stage 1 — Insights
> Bottom-up observations or top-down purposes. Multiple allowed.
>
> ```dataview
> TABLE WITHOUT ID
>   file.link AS "Insight",
>   status AS "Status",
>   tldr AS "Trigger"
> FROM "03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Notes/Settings/Insights"
> WHERE type = "proj-insight"
> SORT created ASC
> ```

> [!info]- Stage 2 — Setting (Locked)
> Formal task / scenario / evaluation / constraints. Single file.
>
> ![[Setting]]

> [!info]- Stage 3 — Paper Review
> Classified by method/approach. Method discovery + similar-work search.
>
> See [[Review-MOC]] for the manual grouping.
>
> ```dataview
> TABLE WITHOUT ID
>   file.link AS "Paper",
>   method_group AS "Group",
>   verdict AS "Verdict",
>   tldr AS "TL;DR"
> FROM "03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Notes/Settings/Paper-Review"
> WHERE type = "proj-paper-note"
> SORT method_group ASC, file.name ASC
> ```

> [!info]- Stage 5.1 — Experiment Logs
> Single-run records. Each log links back to its parent idea.
>
> ```dataview
> TABLE WITHOUT ID
>   file.link AS "Run",
>   idea AS "Idea",
>   status AS "Status",
>   tldr AS "Objective"
> FROM "03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Notes/Experiments"
> WHERE type = "proj-explog"
> SORT created DESC
> ```

---

## Folder Map (created on demand)

```
Semantic-Path-Aware-DDI-Cold-Start/
├── README.md                   ← this file (dashboard + registry)
├── Code/  Notebooks/  Paper/
└── Notes/
    ├── Settings/
    │   ├── Insights/           ← TPL-Proj-Insight
    │   ├── Setting.md          ← TPL-Proj-Setting
    │   └── Paper-Review/       ← TPL-Proj-PaperReview + TPL-Proj-PaperNote
    ├── Ideas/                  ← TPL-Proj-Idea
    └── Experiments/            ← TPL-Proj-ExpLog
```
