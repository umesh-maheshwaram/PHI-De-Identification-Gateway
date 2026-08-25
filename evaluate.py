"""
evaluate.py
===========
The eval harness the brief actually asks for (P2.7): baselines to beat
are (a) regex-only and (b) Presidio/spaCy, scored on entity-level P/R/F1,
leak rate, and (per P2.5) recall reported prominently since a false
negative is a data breach and a false positive is an inconvenience.

Three eval sets are scored SEPARATELY, not pooled, because they answer
different questions:

  1. EVAL_TEMPLATES (synthetic, disjoint from training templates)
     -> "did the model generalize past the exact sentences it trained on?"
  2. WHITFIELD_GOLD (real prose, hand-labeled, from an unrelated document)
     -> "does it work on genuinely out-of-distribution real text?" -
        this is the number to trust most.
  3. Leak rate is computed across both: % of documents where the model
     missed at least one gold identifier entirely. This is reported
     because it's "the number a compliance officer actually asks for"
     (P2.7), and it can be bad even when aggregate F1 looks fine, since
     F1 doesn't distinguish "missed 1 entity in 1 easy doc" from "missed
     1 entity in every doc."

Four detectors are compared, not three: the two required baselines
((a) regex-only, (b) Presidio), the fine-tuned model alone, AND the
ensemble detector (regex claims first, model fills gaps) that
gateway.py actually ships as the production detector. The model-alone
number and the ensemble number are reported side by side deliberately -
see the module docstring in gateway.py for why the ensemble exists: on
real-world text (Whitfield gold), regex-friendly structured categories
(SSN, MRN, dates, IDs) are things a fine-tuned classifier can still miss
that a hand-written pattern catches near-perfectly, so leak rate should
visibly improve from model-only -> ensemble. If it doesn't, that's a
finding worth writing up in FAILURES.md, not a reason to hide the
comparison.

Usage:
    python -m phi_deid.evaluate --model_dir ./phi_deid_model
    python -m phi_deid.evaluate --model_dir ./phi_deid_model --n_synthetic_eval 500
    python -m phi_deid.evaluate --model_dir ./phi_deid_model --no_ensemble
"""
import argparse
from collections import defaultdict

try:
    # Correct usage: `python -m phi_deid.evaluate` (or phi_data.evaluate)
    # run from the PARENT folder.
    from .data_gen import generate_dataset
    from .whitfield_gold import WHITFIELD_GOLD_EXAMPLES
    from .regex_baseline import regex_detect
    from .gateway import build_ensemble_detector
except ImportError:
    # Fallback: `python evaluate.py` run directly from inside the package
    # folder (same situation train.py handles - see its comment for why).
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from data_gen import generate_dataset
    from whitfield_gold import WHITFIELD_GOLD_EXAMPLES
    from regex_baseline import regex_detect
    from gateway import build_ensemble_detector
    print("[evaluate.py] NOTE: running as a loose script, not as a module.\n"
          "  Recommended: cd one level up and run\n"
          "      python -m phi_deid.evaluate --model_dir ./phi_deid_model")


# --------------------------------------------------------------------- #
# Span-level scoring utilities (exact-match entity spans, not token-level
# BIO tags - this is the metric a compliance reviewer actually cares
# about: "did you catch this whole identifier", not "did you get the
# right tag on this one wordpiece").
# --------------------------------------------------------------------- #
def score_spans(gold_spans, pred_spans):
    """gold_spans / pred_spans: list of (start, end, label).
    Returns per-label and overall TP/FP/FN counts using exact span+label
    match. (A stricter but simpler standard than partial-overlap credit;
    partial credit would let a model that only catches 'Whitfield' out of
    'Whitfield, Marcus D.' look better than it should for a leak-rate
    metric where partial redaction can still leave PHI exposed.)"""
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)
    tp = gold_set & pred_set
    fp = pred_set - gold_set
    fn = gold_set - pred_set
    return tp, fp, fn


def aggregate_prf(all_tp, all_fp, all_fn):
    tp, fp, fn = len(all_tp), len(all_fp), len(all_fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def per_label_prf(all_tp, all_fp, all_fn):
    by_label = defaultdict(lambda: {"tp": set(), "fp": set(), "fn": set()})
    for kind, spans in (("tp", all_tp), ("fp", all_fp), ("fn", all_fn)):
        for span in spans:
            by_label[span[2]][kind].add(span)
    rows = {}
    for label in sorted(by_label):
        rows[label] = aggregate_prf(by_label[label]["tp"],
                                     by_label[label]["fp"],
                                     by_label[label]["fn"])
    return rows


def leak_rate(per_doc_results):
    """% of documents containing at least one FALSE NEGATIVE (a missed
    identifier that would have leaked into the foundation LLM call)."""
    n_docs = len(per_doc_results)
    n_leaked = sum(1 for tp, fp, fn in per_doc_results if len(fn) > 0)
    return n_leaked / n_docs if n_docs else 0.0


# --------------------------------------------------------------------- #
# Detector adapters: every detector below exposes the same interface,
# detect(text) -> list[(start, end, label)], so the scoring loop is
# detector-agnostic.
# --------------------------------------------------------------------- #
def regex_adapter(text):
    return [(s, e, lbl) for s, e, lbl in regex_detect(text)]


def make_model_adapter(model_dir):
    """Loads the fine-tuned token-classification model and wraps it as a
    detect(text) -> spans function using the HF token-classification
    pipeline's aggregation_strategy='max' to merge wordpieces back into
    whole entity spans automatically."""
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                               pipeline)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer,
                    aggregation_strategy="max",
                    device=0 if _cuda_available() else -1)

    def detect(text):
        results = ner(text)
        spans = []
        for r in results:
            # pipeline strips the 'entity_group' from 'B-/I-' prefixes and
            # merges contiguous subwords already
            spans.append((int(r["start"]), int(r["end"]), r["entity_group"]))
        return spans

    return detect


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def make_presidio_adapter():
    """Baseline (b) from P2.7. Returns None if presidio isn't installed,
    so evaluate.py degrades gracefully rather than hard-failing."""
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        return None
    analyzer = AnalyzerEngine()

    # Presidio's default entity vocabulary doesn't map 1:1 onto our HIPAA
    # label set (e.g. it has PERSON, not NAME; LOCATION exists both ways).
    # We normalize the labels it CAN detect and leave everything else as
    # a coverage gap, which is itself a fair thing to report - Presidio
    # is a general PII tool, not a HIPAA Safe Harbor tool, and part of
    # what we're demonstrating is that gap.
    PRESIDIO_TO_OURS = {
        "PERSON": "NAME", "LOCATION": "LOCATION", "DATE_TIME": "DATE",
        "PHONE_NUMBER": "PHONE", "EMAIL_ADDRESS": "EMAIL",
        "US_SSN": "SSN", "URL": "URL", "IP_ADDRESS": "IP",
        "MEDICAL_LICENSE": "LICENSE", "US_BANK_NUMBER": "ACCOUNT",
    }

    def detect(text):
        results = analyzer.analyze(text=text, language="en")
        spans = []
        for r in results:
            our_label = PRESIDIO_TO_OURS.get(r.entity_type)
            if our_label:
                spans.append((r.start, r.end, our_label))
        return spans

    return detect


# --------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------- #
def evaluate_detector(name, detect_fn, examples):
    all_tp, all_fp, all_fn = [], [], []
    per_doc_results = []
    for ex in examples:
        gold = ex["entities"]
        pred = detect_fn(ex["text"])
        tp, fp, fn = score_spans(gold, pred)
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)
        per_doc_results.append((tp, fp, fn))

    overall = aggregate_prf(all_tp, all_fp, all_fn)
    overall["leak_rate"] = leak_rate(per_doc_results)
    by_label = per_label_prf(all_tp, all_fp, all_fn)

    print(f"\n=== {name} ===")
    print(f"  Overall  precision={overall['precision']:.3f}  "
          f"recall={overall['recall']:.3f}  (RECALL is the headline "
          f"number - see P2.7)  f1={overall['f1']:.3f}")
    print(f"  Leak rate (docs with >=1 missed identifier): "
          f"{overall['leak_rate']*100:.1f}%  "
          f"({sum(1 for _,_,fn in per_doc_results if fn)}/{len(examples)} docs)")
    print(f"  TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}")
    if by_label:
        print(f"  {'label':12s} {'P':>6s} {'R':>6s} {'F1':>6s} {'support':>8s}")
        for label, m in by_label.items():
            support = m["tp"] + m["fn"]
            print(f"  {label:12s} {m['precision']:6.2f} {m['recall']:6.2f} "
                  f"{m['f1']:6.2f} {support:8d}")
    return overall


def print_fn_examples(name, detect_fn, examples, max_examples=8):
    """Print the actual missed identifiers (false negatives) - the
    concrete evidence behind the leak-rate number, useful for the
    'we will read your masked output looking for leaks' acceptance test
    (P2.9)."""
    print(f"\n--- {name}: sample missed identifiers (false negatives) ---")
    shown = 0
    for ex in examples:
        gold = ex["entities"]
        pred = detect_fn(ex["text"])
        _, _, fn = score_spans(gold, pred)
        for s, e, lbl in sorted(fn):
            print(f"  MISSED [{lbl}] {ex['text'][s:e]!r}  "
                  f"(source: {ex.get('source', 'synthetic')})")
            shown += 1
            if shown >= max_examples:
                return
    if shown == 0:
        print("  (none in this sample)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./phi_deid_model",
                         help="Path to the fine-tuned model saved by train.py")
    parser.add_argument("--n_synthetic_eval", type=int, default=300)
    parser.add_argument("--skip_model", action="store_true",
                         help="Only run the regex/Presidio baselines, "
                              "e.g. if the model hasn't been trained yet")
    parser.add_argument("--no_ensemble", action="store_true",
                         help="Skip the regex+model ensemble comparison "
                              "and only score the model alone. The "
                              "ensemble is what gateway.py actually ships "
                              "in production, so leaving it in is the "
                              "default.")
    args = parser.parse_args()

    synthetic_eval = generate_dataset(args.n_synthetic_eval, seed=777, split="eval")
    whitfield_eval = WHITFIELD_GOLD_EXAMPLES

    detectors = [("Regex-only baseline", regex_adapter)]

    presidio_fn = make_presidio_adapter()
    if presidio_fn:
        detectors.append(("Presidio baseline", presidio_fn))
    else:
        print("[evaluate.py] presidio-analyzer not installed - skipping "
              "baseline (b). `pip install presidio-analyzer presidio-anonymizer "
              "spacy` and `python -m spacy download en_core_web_sm` to include it.")

    if not args.skip_model:
        try:
            model_fn = make_model_adapter(args.model_dir)
            detectors.append(("Fine-tuned DistilBERT alone (ours)", model_fn))

            if not args.no_ensemble:
                # Reuses the already-loaded model pipeline (model_fn)
                # rather than loading the checkpoint a second time -
                # regex claims spans first, model only fills gaps. This
                # is the detector gateway.py actually ships, so it's the
                # number that matters most for P2.7/P2.9, not the
                # model-alone row above.
                ensemble_fn = build_ensemble_detector(model_detect_fn=model_fn)
                detectors.append(("Ensemble regex+model (ours, production)",
                                  ensemble_fn))
        except OSError:
            print(f"[evaluate.py] No trained model found at {args.model_dir} - "
                  f"run train.py first, or pass --skip_model to compare "
                  f"baselines only.")

    print("\n" + "#" * 70)
    print("# EVAL SET 1: synthetic, disjoint EVAL_TEMPLATES "
          "(in-distribution, held out)")
    print("#" * 70)
    for name, fn in detectors:
        evaluate_detector(name, fn, synthetic_eval)

    print("\n" + "#" * 70)
    print("# EVAL SET 2: real Whitfield document excerpts "
          "(genuinely out-of-distribution - trust this number most)")
    print("#" * 70)
    for name, fn in detectors:
        evaluate_detector(name, fn, whitfield_eval)
        print_fn_examples(name, fn, whitfield_eval)


if __name__ == "__main__":
    main()