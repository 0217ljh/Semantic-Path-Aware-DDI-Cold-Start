"""Screen 1: TAG (Text-Attributed Graph) node init for EmerGNN backbone.

Tests whether replacing EmerGNN's Morgan-FP / random init with biomedical
LM-derived node features (PubMedBERT, Qwen-72B input-embedding) improves
cold-start DDI prediction.

See: Notes/Experiments/first_step_plan.md, section 4.5.
"""
