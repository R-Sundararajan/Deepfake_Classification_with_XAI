# Deepfake Classification Research Repository

The current ResNet-18 review prototype runs directly from the repository root. It provides REAL/FAKE classification, model confidence, Grad-CAM, an honest text explanation, evaluation metrics, and a Streamlit interface. See [`system_workflow.md`](system_workflow.md) for measured results and exact commands.

Shared resources remain at the repository root:

- `data/` — existing 140K Real and Fake Faces images (unchanged)
- `datasets/` — dataset setup utilities
- `dfenv/` — existing Python environment
- `models/`, `training/`, `evaluation/`, `inference/`, `explainability/`, `outputs/` — active root-level system
- `old_prototype/` — complete preserved snapshot of the previous prototype, including its legacy ViT implementation
