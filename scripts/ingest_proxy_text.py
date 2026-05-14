"""
Normalize public proxy corpora into UtteranceRecord JSONL / Hugging Face Dataset.

Examples:
  python scripts/ingest_proxy_text.py --source tweet_eval_hate --output data/proxy/tweet_eval.parquet
  python scripts/ingest_proxy_text.py --source synthetic --output data/proxy/synthetic.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset, DatasetDict, load_dataset  # noqa: E402

from hf_ml_verbal.label_config import LABEL_NAMES  # noqa: E402
from hf_ml_verbal.schemas import UtteranceRecord  # noqa: E402


def _stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{h}"


def _tweet_eval_to_records(split: str) -> Dataset:
    ds = load_dataset("tweet_eval", "hate", split=split)
    records: list[dict] = []

    def row_to_labels(label: int) -> dict[str, int]:
        # tweet_eval hate: coarse — map positive hate -> insult_humiliation / explicit_threat proxy
        if label == 1:
            return {
                "neutral_conflict": 0,
                "insult_humiliation": 1,
                "explicit_threat": 0,
                "coercion_harassment": 0,
            }
        return {
            "neutral_conflict": 1,
            "insult_humiliation": 0,
            "explicit_threat": 0,
            "coercion_harassment": 0,
        }

    for row in ds:
        text = row["text"]
        uid = _stable_id("tweet_eval", split + "::" + text)
        rec = UtteranceRecord(
            example_id=uid,
            text=text,
            lang_primary="en",
            lang_tags=["en"],
            source_id="tweet_eval:hate",
            license_spdx="See dataset authors",
            split=split if split != "validation" else "validation",
            labels=row_to_labels(int(row["label"])),
            meta={"raw_label": int(row["label"])},
        )
        records.append(rec.model_dump())

    return Dataset.from_list(records)


def _imdb_smoke(split: str, max_samples: int) -> Dataset:
    ds = load_dataset("imdb", split=split)
    records: list[dict] = []
    for row in ds:
        if len(records) >= max_samples:
            break
        text = row["text"]
        uid = _stable_id("imdb", text)
        rec = UtteranceRecord(
            example_id=uid,
            text=text[:2000],
            lang_primary="en",
            lang_tags=["en"],
            source_id="imdb:smoke",
            license_spdx="See HF imdb dataset card",
            split=split,
            labels={
                "neutral_conflict": 1,
                "insult_humiliation": 0,
                "explicit_threat": 0,
                "coercion_harassment": 0,
            },
            meta={"note": "All mapped to neutral_conflict for pipeline smoke only"},
        )
        records.append(rec.model_dump())
    return Dataset.from_list(records)


def _synthetic_jsonl(path: Path) -> Dataset:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rec = UtteranceRecord(**obj)
        rows.append(rec.model_dump())
    return Dataset.from_list(rows)


def build_dataset(source: str, imdb_cap: int) -> DatasetDict | Dataset:
    if source == "tweet_eval_hate":
        splits = DatasetDict()
        for s in ["train", "validation", "test"]:
            splits[s] = _tweet_eval_to_records(s)
        return splits
    if source == "imdb_smoke":
        return DatasetDict(
            {
                "train": _imdb_smoke("train", imdb_cap),
                "validation": _imdb_smoke("test", max(16, imdb_cap // 4)),
            }
        )
    if source == "synthetic":
        fp = ROOT / "fixtures" / "synthetic_utterances.jsonl"
        full = _synthetic_jsonl(fp)

        def _split_bucket(v: object) -> str:
            return str(v or "").strip().lower()

        train_rows = [
            dict(r)
            for r in full
            if _split_bucket(r["split"]) not in {"validation", "val", "test"}
        ]
        val_rows = [
            dict(r) for r in full if _split_bucket(r["split"]) in {"validation", "val"}
        ]
        if not val_rows:
            split = full.train_test_split(test_size=min(128, max(1, len(full) // 4)), seed=42)
            return DatasetDict({"train": split["train"], "validation": split["test"]})
        test_rows = [dict(r) for r in full if _split_bucket(r["split"]) == "test"]
        dct: dict[str, Dataset] = {}
        if train_rows:
            dct["train"] = Dataset.from_list(train_rows)
        if val_rows:
            dct["validation"] = Dataset.from_list(val_rows)
        if test_rows:
            dct["test"] = Dataset.from_list(test_rows)
        return DatasetDict(dct)

    raise ValueError(f"Unknown source: {source}")


def validate_labels(ds: Dataset) -> None:
    for row in ds:
        missing = set(LABEL_NAMES) - set(row["labels"])
        if missing:
            raise ValueError(f"Missing labels {missing} in {row['example_id']}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=("tweet_eval_hate", "imdb_smoke", "synthetic"),
        default="synthetic",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--imdb-cap", type=int, default=512)
    ap.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Если указано имя типа username/dataset-verbal-proxy — выгрузить Dataset",
    )
    ap.add_argument("--private", action="store_true", help="Создать приватный dataset на Hub")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dsd = build_dataset(args.source, args.imdb_cap)
    if isinstance(dsd, DatasetDict):
        for _k, ds in dsd.items():
            validate_labels(ds)
        dsd.save_to_disk(str(args.output))
        print(f"Saved DatasetDict to {args.output}")
    else:
        validate_labels(dsd)
        dsd.save_to_disk(str(args.output))

    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub, private=args.private)


if __name__ == "__main__":
    main()
