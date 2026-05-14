"""
Fine-tune a multilingual Transformer for multi-label verbal risk classification.

Example:
  python training/train_verbal_classifier.py ^
    --dataset-path data/processed/proxy/synthetic ^
    --output-dir checkpoints/verbal-distil ^
    --epochs 3

При --improve-max-rounds > 1: один прогон train до epochs×rounds эпох с EarlyStopping по eval macro_f1 (без
повторных resume на диске); по умолчанию сохраняются только веса модели — меньше ошибок torch.save оптимизатора под Windows:

  python training/train_verbal_classifier.py ^
    --dataset-path data/processed/proxy/synthetic ^
    --output-dir checkpoints/verbal-iter ^
    --epochs 2 ^
    --improve-max-rounds 8 ^
    --improve-patience 3 ^
    --improve-min-delta 0.005

Полные чекпоинты (оптимизатор на диске): добавьте флаг --save-full-checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from hf_ml_verbal.label_config import ID2LABEL, LABEL2ID, LABEL_NAMES  # noqa: E402


def iteration_log_from_trainer_history(trainer: Trainer) -> list[dict[str, float | int | None]]:
    """Сводка macro_f1 по эпохам из log_history после обучения."""
    out: list[dict[str, float | int | None]] = []
    for row in getattr(trainer.state, "log_history", []) or []:
        metric_key = None
        if "eval_macro_f1_report" in row:
            metric_key = "eval_macro_f1_report"
        elif "eval_macro_f1" in row:
            metric_key = "eval_macro_f1"
        if metric_key is None:
            continue
        try:
            f1 = float(row[metric_key])
        except (TypeError, ValueError):
            continue
        out.append({"epoch": row.get("epoch"), "macro_f1": f1})
    return out


def sanitize_for_json(obj: object):  # noqa: ANN401
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


@dataclass
class EvalSlices:
    overall: dict
    lang_primary_slices: dict


def encode_multilabel_row(example: dict) -> dict:
    lab = example["labels"]
    if not isinstance(lab, dict):
        raise TypeError(f"Expected labels dict on example {example.get('example_id')}")
    return {"labels": [float(lab[name]) for name in LABEL_NAMES]}


def preprocess_batch(tokenizer, max_length: int):
    def _fn(batch: dict) -> dict:
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = batch["labels"]
        return tokens

    return _fn


def evaluate_multilabel(
    y_true: np.ndarray, probs: np.ndarray, langs: np.ndarray | None = None
) -> EvalSlices:
    y_pred = (probs >= 0.5).astype(np.int32)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    def pr_auc_per_label(yt: np.ndarray, pr: np.ndarray) -> tuple[dict[str, float], float]:
        aucs: dict[str, float] = {}
        for i, name in enumerate(LABEL_NAMES):
            yt_i = yt[:, i]
            pr_i = pr[:, i]
            if np.unique(yt_i).size < 2:
                aucs[name] = float("nan")
                continue
            try:
                aucs[name] = float(average_precision_score(yt_i, pr_i))
            except ValueError:
                aucs[name] = float("nan")
        vals = [v for v in aucs.values() if not np.isnan(v)]
        macro_pr = float(np.mean(vals)) if vals else float("nan")
        return aucs, macro_pr

    per_label_ap, macro_pr = pr_auc_per_label(y_true, probs)
    overall = {
        "macro_f1": float(macro_f1),
        "macro_pr_auc": macro_pr,
        "per_label_pr_auc": per_label_ap,
    }

    slices: dict[str, dict[str, float]] = {}
    if langs is None:
        return EvalSlices(overall=overall, lang_primary_slices=slices)

    uniq = sorted(set(map(str, langs)))
    for lang_key in uniq:
        mask = langs == lang_key
        if mask.sum() < 3:
            continue
        yt = y_true[mask]
        pp = probs[mask]
        lf1 = f1_score(yt, (pp >= 0.5).astype(np.int32), average="macro", zero_division=0)
        _, mpr = pr_auc_per_label(yt, pp)
        slices[lang_key] = {"macro_f1": float(lf1), "macro_pr_auc": mpr}

    return EvalSlices(overall=overall, lang_primary_slices=slices)


def make_compute_metrics(eval_label_ids: np.ndarray, eval_langs: np.ndarray | None):
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        probs = 1 / (1 + np.exp(-logits))
        yt = eval_label_ids.astype(np.float32)
        res = evaluate_multilabel(yt, probs, eval_langs)
        flat = dict(res.overall)
        flat["macro_f1_report"] = flat["macro_f1"]
        for name, auc in flat["per_label_pr_auc"].items():
            flat[f"pr_auc__{name}"] = auc
        return flat

    return compute_metrics


def render_model_card(
    hub_placeholder: str, metrics: EvalSlices, readme_path: Path, base_model_name: str
) -> None:
    tmpl = (ROOT / "training" / "MODEL_CARD_TEMPLATE.md").read_text(encoding="utf-8")
    tmpl = tmpl.replace("__BASE_MODEL_PLACEHOLDER__", base_model_name)
    tmpl = tmpl.replace("__METRICS_JSON__", json.dumps(sanitize_for_json(asdict(metrics)), indent=2, ensure_ascii=False, allow_nan=False))
    tmpl = tmpl.replace("__MODEL_REPO__", hub_placeholder)
    readme_path.write_text(tmpl, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=Path, required=True)
    ap.add_argument(
        "--model-name",
        default="distilbert-base-multilingual-cased",
    )
    ap.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/verbal-mbert")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--push-to-hub-model", default=None)
    ap.add_argument(
        "--hub-model-id-readme-placeholder",
        default="username/verbal-multilabel-mbert-placeholder",
    )
    ap.add_argument(
        "--improve-max-rounds",
        type=int,
        default=1,
        metavar="R",
        help="Верхняя граница эпох (= --epochs × это значение): раньше остановит EarlyStopping.",
    )
    ap.add_argument(
        "--improve-patience",
        type=int,
        default=3,
        help="Столько подряд проверок eval без роста metric (по эпохам при eval_strategy=epoch).",
    )
    ap.add_argument(
        "--improve-min-delta",
        type=float,
        default=0.002,
        help="Минимальный прирост eval macro_f1 (см. EarlyStoppingCallback.threshold).",
    )
    ap.add_argument(
        "--save-full-checkpoints",
        action="store_true",
        help="Сохранять оптимизатор и scheduler в checkpoint (тяжелее; под Windows чаще ошибки torch.save).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.improve_max_rounds < 1:
        raise SystemExit("--improve-max-rounds должно быть >= 1")

    ds = load_from_disk(str(args.dataset_path))
    if "train" not in ds:
        raise SystemExit('DatasetDict должен содержать split "train"')

    raw_train = ds["train"].map(encode_multilabel_row)

    eval_split = (
        "validation"
        if "validation" in ds
        else ("test" if "test" in ds else None)
    )
    if eval_split is None:
        n_eval = max(64, min(256, raw_train.num_rows))
        raw_eval_sub = raw_train.shuffle(42).select(range(n_eval))
    else:
        raw_eval_sub = ds[eval_split].map(encode_multilabel_row)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_NAMES),
        id2label={str(i): ID2LABEL[i] for i in range(len(LABEL_NAMES))},
        label2id=LABEL2ID,
        problem_type="multi_label_classification",
    )

    tokenized_train = raw_train.map(
        preprocess_batch(tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_train.column_names,
    )

    tokenized_eval = raw_eval_sub.map(
        preprocess_batch(tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_eval_sub.column_names,
    )

    arr_y = np.array(tokenized_eval["labels"], dtype=np.float32)
    if "lang_primary" in raw_eval_sub.column_names:
        lang_arr = np.array(raw_eval_sub["lang_primary"], dtype=str)
        lang_arr = np.where(lang_arr == "None", "", lang_arr)
    else:
        lang_arr = None

    best_dir = args.output_dir / "best_macro_f1"
    iterative = args.improve_max_rounds > 1
    save_keep = max(12, args.improve_max_rounds * args.epochs + 4) if iterative else 3
    total_epochs = (
        float(args.epochs * args.improve_max_rounds) if iterative else float(args.epochs)
    )
    save_only = iterative and not args.save_full_checkpoints

    callbacks = []
    if iterative:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=max(1, args.improve_patience),
                early_stopping_threshold=max(0.0, args.improve_min_delta),
            )
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=total_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=save_keep,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1_report",
        greater_is_better=True,
        report_to=["none"],
        save_only_model=save_only,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(arr_y, lang_arr),
        callbacks=callbacks,
    )

    trainer.train()

    iteration_log_hist = iteration_log_from_trainer_history(trainer)
    final_metrics = trainer.evaluate()
    macro_f1_final = float(final_metrics["eval_macro_f1_report"])
    if iteration_log_hist:
        best_so_far = max(float(x["macro_f1"]) for x in iteration_log_hist)
        iteration_log: list = iteration_log_hist
    else:
        best_so_far = macro_f1_final
        iteration_log = [{"epoch": None, "macro_f1": macro_f1_final}]

    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    logits = trainer.predict(tokenized_eval).predictions  # noqa: PT028
    probs = 1 / (1 + np.exp(-np.asarray(logits)))
    slice_metrics = evaluate_multilabel(arr_y, probs, lang_arr)

    slice_metrics.overall["best_eval_macro_f1_across_rounds"] = float(best_so_far)

    base_metrics = sanitize_for_json(asdict(slice_metrics))
    base_metrics["iteration_log"] = sanitize_for_json(iteration_log)
    (args.output_dir / "eval_metrics.json").write_text(
        json.dumps(base_metrics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    render_model_card(
        args.hub_model_id_readme_placeholder,
        slice_metrics,
        args.output_dir / "MODEL_CARD.generated.md",
        base_model_name=args.model_name,
    )

    if best_dir.exists() and any(best_dir.iterdir()):
        export_best = args.output_dir / "exported_best_macro_f1"
        if export_best.exists():
            shutil.rmtree(export_best)
        shutil.copytree(best_dir, export_best)

    if args.push_to_hub_model:
        trainer.model.push_to_hub(args.push_to_hub_model)
        tokenizer.push_to_hub(args.push_to_hub_model)


if __name__ == "__main__":
    main()
