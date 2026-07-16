# NBFNet v1.7 — Training Pipeline Flow (cold-start S2 DDI)

Color-coded flow of the NBFNet v1.7 code path: data/setup, the per-epoch `fit`
loop, the per-batch `_train_epoch`, symmetric pair scoring, and the batched
Bellman-Ford kernel. Dotted "drill in" arrows zoom from a step to its internals.
(v1.71 = same flow + TF32/AMP wrap on `_score_pairs` + eval cadence in `fit`.)

```mermaid
flowchart TD
    classDef setup fill:#D1FAE5,stroke:#059669,color:#064E3B,stroke-width:2px
    classDef loop fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef train fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef score fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef bf fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef dec fill:#FEF9C3,stroke:#CA8A04,color:#713F12,stroke-width:2px

    subgraph S1["1 - Data and Setup"]
        direction TB
        A["run_nbfnet.py"] --> B["PairDataset.from_pkl<br/>800-drug seed42"]
        B --> C["_build_kg_inputs<br/>merged KG - 22049 ent<br/>18 rel + 1 DDI slot"]
        C --> D["setup_graph<br/>eval KG = train_ddi + base_kg"]
        D --> E["init_model<br/>NBFNetDDI + Adam + plateau-LR"]
    end

    E --> F0

    subgraph S2["2 - fit - per epoch"]
        direction TB
        F0["_build_epoch_kg<br/>shuffle_train S2"] --> F1["_sample_negatives<br/>emerging-drug pool"]
        F1 --> F2[["_train_epoch"]]
        F2 --> F3{"eval epoch?"}
        F3 -->|"yes"| F4["_validate<br/>static eval KG"]
        F4 --> F5["scheduler.step<br/>best ckpt - early stop"]
        F5 --> F6{"stop?"}
        F6 -->|"no"| F0
        F3 -->|"no"| F0
    end

    F6 -->|"yes"| Z["load best state<br/>predict_proba test_s2<br/>AUC / AUPRC / NLL"]

    subgraph S3["3 - _train_epoch - per batch"]
        direction TB
        G1["batch = pos + neg"] --> G2["_score_pairs<br/>dispatcher"]
        G2 -->|"S2: mask no-op"| G3[["_score_pairs_batched"]]
        G2 -->|"mask needed"| G4["_score_pairs_amortized<br/>fallback"]
        G3 --> G5["BCEWithLogits<br/>backward - step"]
        G4 --> G5
    end

    F2 -.->|"drill in"| G1

    subgraph S4["4 - symmetric pair scoring"]
        direction TB
        H1["unique src a / src b"] --> H2["hf = encode_from_sources a"]
        H1 --> H3["hr = encode_from_sources b"]
        H2 --> H4["h_sym = hf at b + hr at a<br/>representation-level symmetrize"]
        H3 --> H4
        H4 --> H5["mlp_head concat h_sym and query<br/>=> logit"]
    end

    G3 -.->|"drill in"| H1

    subgraph S5["5 - Bellman-Ford kernel"]
        direction TB
        K1["h0 INDICATOR<br/>h0 source = query"] --> K2["L x NBFLayer"]
        K2 --> K3["msg = h * w_q r<br/>w_q r = W_r q + b_r per layer,rel"]
        K3 --> K4["+ boundary reinject h0"]
        K4 --> K5["PNA aggregate<br/>mean/max/min/std x id/amp/atten"]
        K5 --> K6["ReLU -> next layer"]
        K6 --> K2
    end

    H2 -.->|"drill in"| K1
    H3 -.-> K1

    class A,B,C,D,E setup
    class F0,F1,F2,F4,F5,Z loop
    class F3,F6 dec
    class G1,G2,G3,G4,G5 train
    class H1,H2,H3,H4,H5 score
    class K1,K2,K3,K4,K5,K6 bf
```
