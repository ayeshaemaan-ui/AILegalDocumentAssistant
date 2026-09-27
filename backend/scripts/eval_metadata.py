"""Measure the metadata extractors against CUAD's lawyer-annotated ground truth.

Rule-based extraction is easy to fool yourself about: it looks right on the
document you tuned it on. CUAD ships human answers for Parties, Governing Law
and Document Name across all 510 contracts, so the honest move is to score the
extractors on every one of them and publish the number.

Run:  python -m scripts.eval_metadata [--limit N]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingestion.loaders import load_document  # noqa: E402
from app.ingestion.metadata import (  # noqa: E402
    extract_governing_law,
    extract_parties,
    infer_contract_type,
)


def normalise(value: str) -> str:
    value = re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", value).strip()



JURIS_NOISE = {"state", "of", "the", "province", "commonwealth", "republic", "law", "laws"}


def law_match(predicted: str, truth: str) -> bool:
    """Compare jurisdictions on content words.

    CUAD writes "british columbia canada" where the contract says "the Province
    of British Columbia". Those are the same answer, so scoring on shared
    content words is fairer than exact string equality - while still counting
    "nevada" vs "california" as the miss it is.
    """
    pred_tokens = {t for t in normalise(predicted).split() if t not in JURIS_NOISE}
    truth_tokens = {t for t in normalise(truth).split() if t not in JURIS_NOISE}
    if not pred_tokens or not truth_tokens:
        return False
    return pred_tokens <= truth_tokens or truth_tokens <= pred_tokens


def parse_truth_parties(raw: str) -> list[str]:
    """CUAD writes: 'ACME INC, ("Acme"); BETA LLC ("Beta")'."""
    parts = re.split(r"[;]", str(raw))
    out: list[str] = []
    for part in parts:
        part = re.sub(r"\([^)]*\)", " ", part)          # drop the short name
        part = re.sub(r"\b(?:the\s+)?(?:company|parties|party)\b", " ", part, flags=re.I)
        cleaned = normalise(part)
        if len(cleaned) >= 4:
            out.append(cleaned)
    return out


def party_hit(predicted: list[str], truth: list[str]) -> bool:
    """Credit a hit when any predicted name overlaps any true name.

    Matching is deliberately loose (substring either way, or a shared
    distinctive token) because "ACCURAY INCORPORATED" and "Accuray Inc." are
    the same party and an exact-string metric would call that a miss.
    """
    pred_norm = [normalise(p) for p in predicted]
    for truth_name in truth:
        truth_tokens = [t for t in truth_name.split() if len(t) > 3]
        for pred in pred_norm:
            if not pred:
                continue
            if pred in truth_name or truth_name in pred:
                return True
            pred_tokens = {t for t in pred.split() if len(t) > 3}
            if pred_tokens and any(t in pred_tokens for t in truth_tokens):
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N contracts.")
    args = parser.parse_args()

    import pandas as pd

    settings = get_settings()
    csv_path = settings.cuad_dir / "master_clauses.csv"
    txt_dir = settings.cuad_dir / "full_contract_txt"
    if not csv_path.exists():
        print(f"CUAD not found at {settings.cuad_dir}", file=sys.stderr)
        return 1

    frame = pd.read_csv(csv_path)
    by_stem = {Path(str(r["Filename"])).stem: r for _, r in frame.iterrows()}

    files = sorted(txt_dir.glob("*.txt"))
    if args.limit:
        files = files[: args.limit]

    stats = {
        "total": 0,
        "party_scored": 0, "party_hit": 0, "party_empty": 0,
        "law_scored": 0, "law_hit": 0, "law_abstain": 0,
        "type_scored": 0, "type_hit": 0,
    }
    misses: list[tuple[str, str, str]] = []

    for path in files:
        truth = by_stem.get(path.stem)
        if truth is None:
            continue
        try:
            doc = load_document(path)
        except Exception:
            continue
        stats["total"] += 1
        head = doc.full_text[:8000]

        # ---- parties -----------------------------------------------------
        truth_parties = parse_truth_parties(truth.get("Parties-Answer", "") or "")
        if truth_parties:
            stats["party_scored"] += 1
            predicted = extract_parties(head)
            if not predicted:
                stats["party_empty"] += 1
            elif party_hit(predicted, truth_parties):
                stats["party_hit"] += 1
            elif len(misses) < 12:
                misses.append(("parties", str(predicted)[:70], str(truth_parties)[:70]))

        # ---- governing law -----------------------------------------------
        truth_law = normalise(truth.get("Governing Law-Answer", "") or "")
        if truth_law:
            stats["law_scored"] += 1
            predicted_law = extract_governing_law(doc.full_text)
            if predicted_law is None:
                stats["law_abstain"] += 1
            elif law_match(predicted_law, truth_law):
                stats["law_hit"] += 1
            elif len(misses) < 12:
                misses.append(("law", str(predicted_law)[:40], truth_law[:40]))

        # ---- contract type -------------------------------------------------
        truth_name = normalise(truth.get("Document Name-Answer", "") or "")
        if truth_name:
            stats["type_scored"] += 1
            predicted_type = normalise(infer_contract_type(doc.title, head))
            head_word = predicted_type.replace(" agreement", "").strip()
            if head_word and head_word in truth_name:
                stats["type_hit"] += 1

    def pct(hit: int, total: int) -> str:
        return f"{hit}/{total} ({hit / total * 100:5.1f}%)" if total else "n/a"

    print(f"\nScored {stats['total']} contracts against CUAD ground truth\n")
    print(f"  Parties        recall  {pct(stats['party_hit'], stats['party_scored'])}"
          f"   (returned nothing: {stats['party_empty']})")
    law_attempted = stats["law_scored"] - stats["law_abstain"]
    print(f"  Governing law  recall  {pct(stats['law_hit'], stats['law_scored'])}")
    print(f"                 precision when it answers  {pct(stats['law_hit'], law_attempted)}"
          f"   (abstained: {stats['law_abstain']})")
    print(f"  Contract type  match   {pct(stats['type_hit'], stats['type_scored'])}")

    if misses:
        print("\n  Sample misses (field | predicted | truth):")
        for field, predicted, truth_value in misses:
            print(f"    {field:8s} | {predicted:70s} | {truth_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
