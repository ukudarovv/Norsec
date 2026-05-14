from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hf_ml_verbal.label_config import LABEL_NAMES

ROOT = Path(__file__).resolve().parents[1]


def load_verbal_model(
    ckpt_dir: Path | str | None = None,
    device: torch.device | str | None = None,
):
    ckpt_dir = Path(ckpt_dir or ROOT / "checkpoints/verbal-latest")
    ckpt_dir = ckpt_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def verbal_scores_batch(
    model,
    tokenizer,
    device: torch.device,
    texts: list[str],
    max_length: int = 128,
) -> tuple[np.ndarray, list[dict]]:
    texts = [(t or "").strip() for t in texts]
    nonempty = [(i, t) for i, t in enumerate(texts) if t]
    zeros_dict = [{name: 0.0 for name in LABEL_NAMES} for _ in texts]
    if not nonempty:
        return np.zeros((len(texts), len(LABEL_NAMES)), dtype=np.float32), zeros_dict

    batch = [t for _, t in nonempty]
    enc = tokenizer(
        batch,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    logits = model(**enc).logits
    probs = torch.sigmoid(logits).float().cpu().numpy()

    out = np.zeros((len(texts), len(LABEL_NAMES)), dtype=np.float32)
    dicts = [{name: 0.0 for name in LABEL_NAMES} for _ in texts]

    idx_map = [i for i, _t in nonempty]
    for row_out, gi in enumerate(idx_map):
        p = probs[row_out].tolist()
        out[gi] = np.array(p, dtype=np.float32)
        dicts[gi] = {LABEL_NAMES[k]: float(p[k]) for k in range(len(LABEL_NAMES))}

    return out, dicts
