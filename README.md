# PHI De-Identification Gateway

Strips HIPAA Safe Harbor identifiers out of clinical text before it reaches a
foundation LLM, and restores them in the LLM's response — without destroying
the clinical meaning the LLM needs to be useful. Built for **Project 2** of
the LezDo TechMed AI/ML internship assessment.

> **The core tension this project solves (P2.1):** redact too little and you
> leak patient data; redact too much and the downstream LLM becomes useless.
> Entity tagging is the easy part — the masking strategy, the ambiguity
> cases, and the round-trip security are the actual assessment.

---

## Architecture

```
┌─────────────────────────── DATA & LABELS ───────────────────────────┐
│  labels.py               ──label set──▶   data_gen.py                │
│  HIPAA label schema                       Synthetic train/eval        │
│  + BIO tags                               templates                  │
└────────────────────────────────────────────────────────────────────┘
                │ BIO_TAGS + weights          │ TRAIN_TEMPLATES
                ▼                             ▼
┌─────────────────────────── MODEL TRAINING ───────────────────────────┐
│                        train.py                                       │
│         Fine-tune DistilBERT (66M params)                             │
│              + LoRA/QLoRA optional (--lora)                           │
│                        │ saves                                        │
│                        ▼                                              │
│              phi_deid_model/ (checkpoint)                             │
└────────────────────────────────────────────────────────────────────┘
                        │ load
                        ▼
┌──────────────────────────── DETECTORS ───────────────────────────────┐
│  regex_baseline.py        Fine-tuned DistilBERT        Presidio/spaCy │
│  Baseline (a):            (loaded from                 Baseline (b), │
│  structured regex          phi_deid_model)              optional      │
└────────────────────────────────────────────────────────────────────┘
        │ baseline a          │ model alone          │ baseline b
        ▼                     ▼                       ▼
┌─────────────────────── EVALUATION HARNESS ───────────────────────────┐
│  whitfield_gold.py  ──fills gaps──▶      evaluate.py                  │
│  Real hand-labeled                       Scores regex / Presidio /    │
│  gold text                               model / ensemble             │
│  (trust this number                      → P/R/F1 + leak rate         │
│   most)                                                                │
└────────────────────────────────────────────────────────────────────┘
                        │ --batch demo
                        ▼
┌────────────────────── PRODUCTION GATEWAY (gateway.py) ───────────────┐
│  build_ensemble_detector()    deidentify() / rehydrate()   Foundation │
│  regex claims first,      ──▶ + pre-send leak self-check──▶ LLM       │
│  model fills gaps             masked_text, mapping          (Groq /   │
│                                                     masked_text only   Anthropic│
│                                                               / Ollama)│
└────────────────────────────────────────────────────────────────────┘
        → rehydrated response returned to caller
          (mapping never leaves the server, never sent to the LLM)
```

*A rendered version of this diagram is included at
[`docs/pipeline_architecture.png`](docs/pipeline_architecture.png).*

### Why this shape, not model-only

- **Detector ensemble, not model-only** (`gateway.build_ensemble_detector`):
  `regex_baseline.py` gets near-perfect recall on structured, high-entropy
  categories (SSN, email, URL, IP, numeric dates, MRN/ACCOUNT/LICENSE-style
  codes) because they have a rigid grammar. The fine-tuned model contributes
  what regex structurally cannot — names, locations, and eponym/facility
  ambiguity (`Dr. Parkinson diagnosed Parkinson's`). Regex claims spans
  first; the model only proposes spans for text regex left unclaimed.
- **Three eval sets are scored separately, not pooled**, because they answer
  different questions: synthetic held-out templates ("did it generalize past
  its own training sentences?"), Whitfield gold ("does it work on genuinely
  out-of-distribution real prose?" — the number to trust most), and leak
  rate across both ("the number a compliance officer actually asks for").
- **The gateway is the actual deliverable for P2.2/P2.3**, not just the
  detector — `evaluate.py` proves detection works; `gateway.py` is the
  callable service (`deidentify`/`rehydrate`) that a real caller would use.

---

## Repository structure

```
phi_deid/
├── README.md                  this file
├── requirements.txt
├── labels.py                  HIPAA Safe Harbor label schema, BIO tags, recall weighting
├── data_gen.py                synthetic training/eval data generator (Faker-based)
├── regex_baseline.py          baseline (a): structured-pattern regex detector
├── train.py                   fine-tunes DistilBERT (full fine-tune or --lora)
├── evaluate.py                scores regex / Presidio / model-alone / ensemble
├── gateway.py                 production service: deidentify / rehydrate / round_trip
├── whitfield_gold.py          hand-labeled real-text gold set (see data/ below)
├── phi_deid_model/             trained checkpoint (or reproduce via train.py)
├── data/
│   └── ...                    supplementary data (see submission email)
├── results/
│   ├── eval_synthetic.txt
│   ├── eval_whitfield_gold.txt
│   ├── gateway_report.md      from `gateway.py --batch --report_out`
│   └── FAILURES.md
└── docs/
    └── pipeline_architecture.png
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Optional (baseline (b), P2.7):
python -m spacy download en_core_web_sm
```

Tested on: Windows, Python 3.10+, NVIDIA GTX 1650 (4GB VRAM). DistilBERT at
66M params trains and serves comfortably on this hardware — no LoRA required
for training feasibility, though `--lora` is supported to explicitly exercise
that path per the hard requirement.

---

## Usage

### 1. Generate synthetic data (used internally by train.py/evaluate.py)

```bash
python data_gen.py            # smoke-test the generator directly
```

### 2. Train the detector

```bash
python train.py --epochs 8 --n_train 8000        # full fine-tune
python train.py --epochs 8 --lora                 # LoRA/QLoRA path
```

Saves the best checkpoint (selected by **recall**, not loss — a missed
identifier is a data breach, a false positive is an inconvenience, per P2.5)
to `./phi_deid_model`.

### 3. Evaluate

```bash
python evaluate.py --model_dir ./phi_deid_model
```

Reports regex-only, Presidio (if installed), model-alone, and the
regex+model **ensemble** (what `gateway.py` actually ships), each scored
separately on the synthetic held-out set and the real Whitfield gold set,
with entity-level P/R/F1 per label, leak rate, and sample missed
identifiers.

### 4. Run the gateway (single document)

```bash
set GROQ_API_KEY=your_key_here
python gateway.py --model_dir ./phi_deid_model --input some_note.txt
```

Prints the raw text, masked text sent to the LLM, the LLM's (still masked)
response, and the rehydrated response, with per-stage timing.

### 5. Run the gateway (batch, over the real Whitfield documents)

```bash
python gateway.py --model_dir ./phi_deid_model --batch --report_out results/gateway_report.md
```

Produces p50/p95/mean/max latency across `deidentify`, `llm_call`, and
`total`, plus a full markdown transcript for the submission.

Other foundation-LLM backends:

```bash
# Local, no API key, nothing leaves the machine:
python gateway.py --provider openai_compatible --llm_model llama3.2:3b

# Anthropic:
set ANTHROPIC_API_KEY=your_key_here
python gateway.py --provider anthropic --llm_model claude-sonnet-4-5
```

---

## Identifier scope (P2.4)

All 18 HIPAA Safe Harbor categories are addressed; coverage is defined in
`labels.py`. Categories 16 (biometric identifiers) and 17 (full-face
photographs) are **explicitly out of scope** — this is a text-only gateway;
they require signal/waveform or image input, which is a separate service.
All other categories (1–15, 18) are covered by the `LABELS` set and mapped
to their HIPAA category number in `ENTITY_TO_HIPAA_CATEGORY`.

## The hard parts (P2.5) — how each is handled

| Challenge | Approach |
|---|---|
| Ambiguity (`Dr. Parkinson` vs `Parkinson's`) | Fine-tuned model, not regex — this is exactly why a trained classifier is needed instead of pattern matching. |
| Dates / interval preservation | Every date in a document is shifted by the **same** random per-document offset, so "8 days after the collision" stays true while calendar dates are de-identified. Unparseable date strings fall back to placeholder masking rather than being left real. |
| Ages > 89 | Collapsed to the fixed token `"90+"`, and deliberately **not** added to the reverse mapping — HIPAA Safe Harbor 3b requires generalizing the age, not hiding-then-restoring it. |
| Recall asymmetry | `RECALL_CRITICAL_WEIGHT` in `labels.py` weights high-risk categories (SSN, MRN, HEALTHPLAN, NAME) more heavily; `evaluate.py` reports recall as the headline metric, not F1. |
| Masking strategy | Consistent pseudonymisation (`[NAME_1]`, `[MRN_1]`, ...) for most categories — every occurrence of the same original string maps to the same placeholder, so the LLM can still reason about "the patient" as one consistent entity. Chosen over pure redaction (destroys clinical meaning) and surrogate generation (risks the LLM treating a fabricated identity as real). |
| Rehydration security | The mapping is held in-memory by the caller and never sent to the LLM. `rehydrate()` only substitutes strings it produced itself; any bracket-shaped token in the LLM's response that ISN'T a mapping key is left untouched and reported as a warning, not silently trusted. |

## Pre-send leak self-check

Before masked text is ever sent to the foundation LLM, `gateway.py` re-runs
`regex_detect()` against the already-masked text. Anything it still finds is
masked again before sending — the automatable half of "we will read your
masked output looking for leaks" (P2.9).

---

## Known limitations

See [`results/FAILURES.md`](results/FAILURES.md) for the full, honestly
reported list, including diagnosed weak categories (FAX, SSN, ACCOUNT on the
synthetic eval) and a self-check false-positive bug found and fixed during
development (shifted dates were briefly being re-flagged as leaks by the
self-check and clobbered back into placeholders).
