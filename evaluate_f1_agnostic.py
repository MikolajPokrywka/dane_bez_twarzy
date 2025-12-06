"""
Evaluate placeholder-level precision/recall/F1 treating all classes as the same.

This evaluation ignores the specific placeholder type (e.g., [name], [city], [phone])
and only checks whether SOME placeholder exists at a given position.

Useful for measuring if the anonymization detects sensitive data at the right locations,
even if the label is incorrect.

Usage:
  python evaluate_f1_agnostic.py --pred out.txt --ref test.ref.txt
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import re
import sys
from typing import Dict, Iterable, List, Tuple

PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")


def extract_spans_agnostic(line: str) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) spans for any placeholders in a line, ignoring the label.

    NOTE: Kept for backwards compatibility and potential debugging, but the main
    evaluation now uses context-based matching that is robust to small shifts
    in character positions.
    """
    return [(m.start(), m.end()) for m in PLACEHOLDER_RE.finditer(line)]


def extract_spans_with_labels(line: str) -> List[Tuple[int, int, str]]:
    """
    Return list of (start, end, label) spans for placeholders in a line.

    NOTE: Kept for backwards compatibility and potential debugging; label-aware
    evaluation below relies on a context-based matcher instead.
    """
    return [(m.start(), m.end(), m.group()) for m in PLACEHOLDER_RE.finditer(line)]


def extract_placeholders_with_context(line: str, context_len: int = 20) -> List[Dict[str, str]]:
    """
    Extract placeholders with short left/right context snippets.

    This is similar to the logic used in the Streamlit comparison tool, but
    implemented here without external dependencies so it can be used for
    automated evaluation.
    """
    placeholders: List[Dict[str, str]] = []
    for m in PLACEHOLDER_RE.finditer(line):
        start, end = m.start(), m.end()
        before = line[max(0, start - context_len) : start]
        after = line[end : min(len(line), end + context_len)]
        placeholders.append(
            {
                "label": m.group(),
                "before": before,
                "after": after,
                "start": str(start),  # not used for matching, but can help debugging
                "end": str(end),
            }
        )
    return placeholders


def match_placeholders_by_context(
    pred_phs: List[Dict[str, str]], ref_phs: List[Dict[str, str]]
) -> Dict[str, List]:
    """
    Match placeholders based on surrounding context, not raw character position.

    This makes the metric robust to small edits (e.g. punctuation, wording)
    that shift spans but keep the sensitive slot in the same textual context.

    Returns:
        {
          "matched": [(pred_idx, ref_idx, label_match_bool), ...],
          "fp_indices": [pred_idx, ...],
          "fn_indices": [ref_idx, ...],
        }
    """
    matched: List[Tuple[int, int, bool]] = []
    pred_matched = set()
    ref_matched = set()

    # Greedy one-to-one matching by best context score
    for i, pred in enumerate(pred_phs):
        best_match = None
        best_score = 0

        for j, ref in enumerate(ref_phs):
            if j in ref_matched:
                continue

            before_match = pred["before"] == ref["before"]
            after_match = pred["after"] == ref["after"]
            score = (2 if before_match else 0) + (1 if after_match else 0)

            if score > best_score:
                best_score = score
                best_match = j

        # Require at least the "before" context to match to avoid spurious pairs
        if best_match is not None and best_score >= 2:
            label_match = pred_phs[i]["label"] == ref_phs[best_match]["label"]
            matched.append((i, best_match, label_match))
            pred_matched.add(i)
            ref_matched.add(best_match)

    fp_indices = [i for i in range(len(pred_phs)) if i not in pred_matched]
    fn_indices = [j for j in range(len(ref_phs)) if j not in ref_matched]

    return {
        "matched": matched,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
    }


def safe_readlines(path: pathlib.Path, encoding: str) -> Iterable[str]:
    """Stream file lines safely with the given encoding."""
    with path.open("r", encoding=encoding, errors="ignore") as f:
        for line in f:
            yield line.rstrip("\n")


def evaluate_agnostic(pred_path: pathlib.Path, ref_path: pathlib.Path, encoding: str = "utf-8") -> None:
    """
    Compute label-agnostic precision/recall/F1.

    The metric is now based on context-aware matching of placeholders rather
    than raw character positions, making it robust to small textual shifts.
    """
    pred_lines = safe_readlines(pred_path, encoding)
    ref_lines = safe_readlines(ref_path, encoding)

    tp = fp = fn = 0

    line_count = 0
    for line_count, (pred_line, ref_line) in enumerate(
        itertools.zip_longest(pred_lines, ref_lines), start=1
    ):
        if pred_line is None or ref_line is None:
            sys.stderr.write(
                f"Warning: line count mismatch; stopping at line {line_count-1}.\n"
            )
            break

        # Context-based matching ignores exact character offsets
        pred_phs = extract_placeholders_with_context(pred_line)
        ref_phs = extract_placeholders_with_context(ref_line)
        matching = match_placeholders_by_context(pred_phs, ref_phs)

        line_tp = len(matching["matched"])
        line_fp = len(matching["fp_indices"])
        line_fn = len(matching["fn_indices"])

        tp += line_tp
        fp += line_fp
        fn += line_fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("=" * 60)
    print("LABEL-AGNOSTIC EVALUATION")
    print("(All placeholder types treated as equivalent)")
    print("=" * 60)
    print(f"Lines compared: {line_count if line_count else 0}")
    print(f"Gold spans: {tp + fn} | Pred spans: {tp + fp}")
    print(f"TP: {tp} FP: {fp} FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("=" * 60)


def evaluate_with_confusion(pred_path: pathlib.Path, ref_path: pathlib.Path, encoding: str = "utf-8") -> None:
    """
    Show label-agnostic metrics plus confusion analysis.

    Uses the same context-based matching as the simple evaluation, then
    refines matches to check label correctness and build a confusion matrix.
    """
    pred_lines = safe_readlines(pred_path, encoding)
    ref_lines = safe_readlines(ref_path, encoding)

    tp_agnostic = fp = fn = 0
    tp_exact = 0  # True positives with correct label
    label_mismatch = 0  # Right position, wrong label
    
    from collections import defaultdict
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    line_count = 0
    for line_count, (pred_line, ref_line) in enumerate(
        itertools.zip_longest(pred_lines, ref_lines), start=1
    ):
        if pred_line is None or ref_line is None:
            sys.stderr.write(
                f"Warning: line count mismatch; stopping at line {line_count-1}.\n"
            )
            break

        # Extract placeholders with context and match them line-by-line
        pred_phs = extract_placeholders_with_context(pred_line)
        ref_phs = extract_placeholders_with_context(ref_line)
        matching = match_placeholders_by_context(pred_phs, ref_phs)

        line_tp_agnostic = len(matching["matched"])
        line_fp = len(matching["fp_indices"])
        line_fn = len(matching["fn_indices"])

        # Among context-matched pairs, count label matches vs mismatches
        line_tp_exact = sum(1 for _, _, lm in matching["matched"] if lm)
        
        tp_agnostic += line_tp_agnostic
        tp_exact += line_tp_exact
        fp += line_fp
        fn += line_fn
        label_mismatch += (line_tp_agnostic - line_tp_exact)

        # Build confusion matrix for context-matched spans with different labels
        for pred_idx, ref_idx, lm in matching["matched"]:
            if lm:
                continue
            pred_label = pred_phs[pred_idx]["label"]
            ref_label = ref_phs[ref_idx]["label"]
            confusion_matrix[ref_label][pred_label] += 1

    precision_agnostic = tp_agnostic / (tp_agnostic + fp) if (tp_agnostic + fp) else 0.0
    recall_agnostic = tp_agnostic / (tp_agnostic + fn) if (tp_agnostic + fn) else 0.0
    f1_agnostic = 2 * precision_agnostic * recall_agnostic / (precision_agnostic + recall_agnostic) if (precision_agnostic + recall_agnostic) else 0.0

    precision_exact = tp_exact / (tp_exact + fp) if (tp_exact + fp) else 0.0
    recall_exact = tp_exact / (tp_exact + fn) if (tp_exact + fn) else 0.0
    f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if (precision_exact + recall_exact) else 0.0

    print("=" * 70)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Lines compared: {line_count if line_count else 0}")
    print(f"\nGold spans: {tp_agnostic + fn} | Pred spans: {tp_agnostic + fp}")
    
    print("\n" + "=" * 70)
    print("LABEL-AGNOSTIC METRICS (position-only)")
    print("=" * 70)
    print(f"TP (any label):  {tp_agnostic}")
    print(f"FP:              {fp}")
    print(f"FN:              {fn}")
    print(f"Precision:       {precision_agnostic:.4f}")
    print(f"Recall:          {recall_agnostic:.4f}")
    print(f"F1:              {f1_agnostic:.4f}")
    
    print("\n" + "=" * 70)
    print("LABEL-AWARE METRICS (exact match)")
    print("=" * 70)
    print(f"TP (exact):      {tp_exact}")
    print(f"Label mismatches:{label_mismatch} ({label_mismatch/tp_agnostic*100:.1f}% of detected)" if tp_agnostic > 0 else "Label mismatches: 0")
    print(f"FP:              {fp}")
    print(f"FN:              {fn}")
    print(f"Precision:       {precision_exact:.4f}")
    print(f"Recall:          {recall_exact:.4f}")
    print(f"F1:              {f1_exact:.4f}")
    
    if confusion_matrix:
        print("\n" + "=" * 70)
        print("LABEL CONFUSION (same position, different label)")
        print("=" * 70)
        print(f"{'Reference':<20} {'Predicted':<20} {'Count':>10}")
        print("-" * 70)
        for ref_label in sorted(confusion_matrix.keys()):
            for pred_label in sorted(confusion_matrix[ref_label].keys()):
                count = confusion_matrix[ref_label][pred_label]
                print(f"{ref_label:<20} {pred_label:<20} {count:>10}")
    
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute label-agnostic F1 between predicted and reference anonymized files."
    )
    parser.add_argument("--pred", required=True, help="Path to predicted anonymized file.")
    parser.add_argument("--ref", required=True, help="Path to reference anonymized file.")
    parser.add_argument("--encoding", default="utf-8", help="Text encoding (default: utf-8).")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Show only label-agnostic metrics (simpler output)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = pathlib.Path(args.pred).expanduser()
    ref_path = pathlib.Path(args.ref).expanduser()

    if not pred_path.exists():
        sys.stderr.write(f"Pred file missing: {pred_path}\n")
        sys.exit(1)
    if not ref_path.exists():
        sys.stderr.write(f"Ref file missing: {ref_path}\n")
        sys.exit(1)

    if args.simple:
        evaluate_agnostic(pred_path, ref_path, args.encoding)
    else:
        evaluate_with_confusion(pred_path, ref_path, args.encoding)


if __name__ == "__main__":
    main()

