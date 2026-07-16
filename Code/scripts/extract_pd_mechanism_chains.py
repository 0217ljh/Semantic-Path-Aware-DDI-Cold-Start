"""Per-sample LLM extraction of PD DDI mechanism as a convergence chain.

Reads the 200 PD pairs with DDInter clinical mechanism text
(`Code/data/_cache/pd_mechanism_200.csv`), and for each one asks Claude to
structure the clinician-written mechanism into an explicit convergence chain:

    drug_A --[action]--> target_A --> [convergence system] <-- target_B <--[action]-- drug_B

plus a mechanism category and whether both targets are molecular (KG-mappable).
The point: get the RELIABLE mechanism per pair from the clinical text, so the
user can then locate target_A / target_B in the KG and flag the missing
convergence node. Source = DDInter text + standard pharmacology, NOT the KG.

Results are cached to JSONL (re-runs skip done pairs). Read-only on data.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/extract_pd_mechanism_chains.py --limit 12
    PYTHONPATH=Code python -u Code/scripts/extract_pd_mechanism_chains.py            # full 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

import anthropic
import pandas as pd

IN_CSV = "Code/data/_cache/pd_mechanism_200.csv"
OUT_DIR = pathlib.Path("Code/runs/pd_mechanism")
OUT_JSONL = OUT_DIR / "chains.jsonl"
KEY_FILE = "API-KEY/API-KEY.txt"

SYSTEM = (
    "You are a clinical pharmacologist extracting drug-drug interaction "
    "mechanisms as structured convergence chains. Output ONLY a single JSON "
    "object and nothing else."
)

USER_TMPL = """Drug A: {a}
Drug B: {b}
Interaction effect: {effect}
Clinical mechanism (DDInter, clinician-written): {mech}

Extract the pharmacodynamic mechanism as a CONVERGENCE chain: each drug acts (via a molecular target or process) to perturb a shared physiological system or endpoint. Use the clinical text plus standard pharmacology.

Output JSON with exactly these keys:
- "a_target": most specific molecular target/process drug A acts on (e.g. "serotonin transporter (SERT)", "hERG (KCNH2)", "GABA-A receptor", "monoamine oxidase"); "" if not identifiable
- "a_action": A's action on it (inhibitor/agonist/antagonist/inducer/blocker/substrate/releaser/...)
- "convergence": the shared physiological node/endpoint BOTH drugs perturb (e.g. "synaptic serotonin / 5-HT1A&2A receptors", "QT interval / cardiac repolarization", "CNS depression", "blood pressure / vascular tone", "serum potassium")
- "b_target": same as a_target, for drug B
- "b_action": same as a_action, for drug B
- "category": short mechanism class (e.g. "serotonergic excess", "additive CNS depression", "additive QT prolongation", "additive anticholinergic", "additive hyperkalemia")
- "chain": one line, exactly "A --[a_action]--> a_target ==> [convergence] <== b_target <--[b_action]-- B" with names filled in
- "molecular_grounded": true if BOTH a_target and b_target are concrete molecular entities (protein/channel/enzyme/transporter) that could be KG nodes; false if the mechanism only exists at a systemic/physiological level
JSON only, no prose."""


def load_key() -> str:
    txt = pathlib.Path(KEY_FILE).read_text(errors="ignore")
    m = re.search(r"sk-ant-[A-Za-z0-9_\-]+", txt)
    if not m:
        sys.exit(f"no sk-ant key found in {KEY_FILE}")
    return m.group()


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all 200")
    ap.add_argument("--model", default="claude-opus-4-8")
    args = ap.parse_args()

    df = pd.read_csv(IN_CSV)
    if args.limit:
        df = df.head(args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r

    client = anthropic.Anthropic(api_key=load_key())
    n_call = 0
    with OUT_JSONL.open("a", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            key = f"{row['drug_a_id']}|{row['drug_b_id']}"
            if key in done:
                continue
            user = USER_TMPL.format(a=row["drug_a_name"], b=row["drug_b_name"],
                                    effect=str(row["ddi_type"])[:90],
                                    mech=str(row["original_text"])[:1400])
            try:
                resp = client.messages.create(
                    model=args.model, max_tokens=700,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": user}])
                text = "".join(b.text for b in resp.content if b.type == "text")
                data = parse_json(text)
            except Exception as e:  # noqa: BLE001
                data, text = None, f"ERROR {e}"
            rec = {"key": key, "a": row["drug_a_name"], "b": row["drug_b_name"],
                   "effect": str(row["ddi_type"])[:60],
                   "kg_chain": str(row["has_key_entity"]).lower() == "true",
                   "parsed": data, "raw": None if data else text[:300]}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[key] = rec
            n_call += 1
    print(f"called LLM {n_call}x | total cached {len(done)}", flush=True)

    # ---- report ----
    recs = [r for r in done.values() if r.get("parsed")]
    print(f"\nparsed {len(recs)}/{len(done)}")
    cat = Counter(r["parsed"].get("category", "?") for r in recs)
    mol = sum(bool(r["parsed"].get("molecular_grounded")) for r in recs)
    print(f"molecular_grounded (both targets are KG-mappable molecules): "
          f"{mol}/{len(recs)} ({mol/max(len(recs),1)*100:.0f}%)")
    print("\n--- mechanism category distribution ---")
    for c, n in cat.most_common(15):
        print(f"  {n:>3}  {c}")
    print("\n--- top convergence nodes ---")
    conv = Counter(r["parsed"].get("convergence", "?") for r in recs)
    for c, n in conv.most_common(12):
        print(f"  {n:>3}  {c}")
    print("\n--- example chains ---")
    for r in recs[:20]:
        p = r["parsed"]
        flag = "MOL" if p.get("molecular_grounded") else "sys"
        print(f"[{flag}] {p.get('chain','')[:140]}")

    # export readable CSV of ALL parsed chains
    rows = []
    for r in recs:
        p = r["parsed"]
        rows.append({"drug_a": r["a"], "drug_b": r["b"], "effect": r["effect"],
                     "kg_shared_target": r["kg_chain"],
                     "a_target": p.get("a_target"), "a_action": p.get("a_action"),
                     "convergence": p.get("convergence"),
                     "b_target": p.get("b_target"), "b_action": p.get("b_action"),
                     "category": p.get("category"),
                     "molecular_grounded": p.get("molecular_grounded"),
                     "chain": p.get("chain")})
    out_csv = OUT_DIR / "chains.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote readable CSV ({len(rows)} rows) -> {out_csv}")


if __name__ == "__main__":
    main()
