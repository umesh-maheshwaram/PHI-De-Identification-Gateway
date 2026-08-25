"""
gateway.py
==========
The actual deliverable for P2.2/P2.3 of the brief: a callable service that
sits in front of a foundation LLM.

    deidentify(text)            -> (masked_text, mapping)
    rehydrate(response, mapping) -> text
    round_trip(text, question)  -> full raw -> masked -> LLM -> rehydrated demo

evaluate.py and train.py already answer "does the detector work". This
file answers "is there a product" - the thing the brief says will be
demoed live on the review call (P2.9).

Design decisions, and why (P2.5 "the hard parts"):

  DETECTOR ENSEMBLE, NOT MODEL-ONLY
    regex_baseline.py's own docstring says it belongs in the production
    gateway, not just as a throwaway baseline - it gets near-perfect
    recall on structured, high-entropy categories (SSN, email, URL, IP,
    numeric dates, MRN/ACCOUNT/LICENSE-style codes) because those have a
    rigid grammar a regex captures better than a fine-tuned classifier
    memorizing surface patterns. The model's fine-tuned distilbert
    contributes what regex structurally cannot: names, locations, and
    the eponym/facility ambiguity cases (P2.5 "Ambiguity"). We run regex
    FIRST and let it claim spans; the model only proposes spans for
    unclaimed text. This is the single highest-leverage lever for
    lowering leak rate on real documents (see FAILURES.md) - it doesn't
    require retraining anything.

  MASKING STRATEGY: consistent pseudonymisation + date shifting
    Redaction ([NAME]) destroys clinical meaning outright (P2.5's
    stated tension: "redact too much and the downstream LLM becomes
    useless"). Surrogate generation (fabricated realistic names) risks
    the LLM treating a fake identity as real and reasoning about it as
    if it were true. We use:
      - Consistent pseudonymisation for NAME/LOCATION/ACCOUNT/etc:
        every occurrence of the same original string within one
        document maps to the same placeholder ("NAME_1" appears
        wherever "Whitfield, Marcus D." appeared), so the LLM can still
        reason about "the patient" as one consistent entity across the
        note.
      - Consistent per-document date shifting for DATE: every date in
        the document is shifted by the SAME random offset, so intervals
        ("3 days post-op", "8 days after the collision") are preserved
        exactly, while the actual calendar dates are de-identified.
        This directly answers P2.5's "Dates" challenge.
      - AGE > 89 collapses to the fixed token "90+", per HIPAA Safe
        Harbor 3b. This is intentionally NOT reversible - Safe Harbor
        requires generalizing the age, not just hiding it, so
        rehydration correctly leaves "90+" as-is rather than restoring
        the real age.

  REHYDRATION SECURITY (P2.5 "Rehydration security")
    The mapping is an in-memory dict returned to and held by the
    caller, keyed by the exact masked string that appears in
    masked_text - it never touches disk and is never sent to the LLM.
    rehydrate() only ever substitutes strings that appear as VALUES it
    itself generated; if the LLM echoes a token that was never in the
    input (hallucinated an ID, or copied a bracket-looking phrase from
    its own training data), rehydrate() will not match it against
    anything in the mapping and will leave it untouched in the output.
    We additionally flag any bracketed [LABEL_n]-shaped token in the
    LLM's response that ISN'T a key in the mapping, since that's exactly
    the "LLM echoes a token that was never in the input" failure mode
    the brief asks about - it gets surfaced as a warning rather than
    silently trusted or silently dropped.

  PRE-SEND LEAK SELF-CHECK
    Before masked_text is ever sent to the foundation LLM, we re-run
    regex_detect() against the ALREADY-MASKED text. If it finds
    anything (e.g. the model missed an SSN that regex would have
    caught, or a placeholder token got mangled), we mask those spans
    too before sending, and report them. This is the automatable half
    of "we will read your masked output looking for leaks" (P2.9) - it
    can't catch everything a human reviewer would, but it catches the
    cheap, structured, unambiguous stuff for free.
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timedelta

try:
    from .labels import LABELS
    from .regex_baseline import regex_detect
    from .whitfield_gold import WHITFIELD_GOLD_EXAMPLES
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from labels import LABELS
    from regex_baseline import regex_detect
    from whitfield_gold import WHITFIELD_GOLD_EXAMPLES
    print("[gateway.py] NOTE: running as a loose script, not as a module.\n"
          "  Recommended: cd one level up and run\n"
          "      python -m phi_deid.gateway --demo")


# --------------------------------------------------------------------- #
# Date parsing / shifting - P2.5 "Dates"
# --------------------------------------------------------------------- #
_DATE_FORMATS = ["%m/%d/%Y", "%m/%d/%y", "%B %d, %Y"]


def _parse_date(text: str):
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt), fmt
        except ValueError:
            continue
    return None, None


def shift_date(text: str, offset_days: int):
    """Shift a date string by offset_days, preserving its original
    format. Returns None if the string doesn't match a known date
    format (caller falls back to placeholder masking in that case,
    rather than silently leaving an unparseable-but-real date in the
    output)."""
    dt, fmt = _parse_date(text)
    if dt is None:
        return None
    shifted = dt + timedelta(days=offset_days)
    return shifted.strftime(fmt)


# --------------------------------------------------------------------- #
# Detector ensemble
# --------------------------------------------------------------------- #
def _spans_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def make_model_detector(model_dir):
    """Loads the fine-tuned token-classification model, same loading
    path as evaluate.py's make_model_adapter, kept independent here so
    gateway.py has no import-time dependency on evaluate.py."""
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                               pipeline)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except ImportError:
        device = -1
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer,
                    aggregation_strategy="max", device=device)

    def detect(text):
        return [(int(r["start"]), int(r["end"]), r["entity_group"])
                for r in ner(text)]

    return detect


def build_ensemble_detector(model_dir=None, model_detect_fn=None):
    """Regex claims spans first (near-perfect on structured categories -
    see module docstring); the model only proposes spans for text regex
    left unclaimed. Returns detect(text) -> list[(start, end, label)],
    sorted, non-overlapping."""
    model_detect = model_detect_fn or (make_model_detector(model_dir) if model_dir else None)

    def detect(text):
        regex_spans = sorted(regex_detect(text))
        claimed = [False] * (len(text) + 1)
        for s, e, _ in regex_spans:
            for i in range(s, e):
                claimed[i] = True

        combined = list(regex_spans)
        if model_detect is not None:
            for s, e, label in sorted(model_detect(text)):
                if any(claimed[s:e]):
                    continue  # regex already owns this region
                combined.append((s, e, label))
                for i in range(s, e):
                    claimed[i] = True

        combined.sort(key=lambda x: x[0])
        return combined

    return detect


# --------------------------------------------------------------------- #
# deidentify / rehydrate
# --------------------------------------------------------------------- #
def _leak_selfcheck(masked_text):
    """Re-run regex against ALREADY-MASKED text. Anything it still finds
    is a leak that slipped through - see module docstring."""
    return regex_detect(masked_text)


def deidentify(text: str, detect_fn, date_shift_days: int = None,
               run_selfcheck: bool = True):
    """Returns (masked_text, mapping).

    mapping = {
        "map": {masked_string: original_string, ...},   # for rehydrate()
        "date_shift_days": int,                          # audit trail
        "spans_found": [(start, end, label), ...],       # audit trail
        "selfcheck_leaks": [(start, end, label), ...],   # should be []
    }

    mapping is meant to be held by the CALLER (e.g. kept server-side,
    per-request, never persisted alongside masked_text) and passed back
    into rehydrate(). It is never sent to the foundation LLM.
    """
    if date_shift_days is None:
        date_shift_days = random.randint(-180, 180) or 1  # never 0

    spans = detect_fn(text)
    counters = {}          # label -> next placeholder index
    memo = {}              # (label, original.lower()) -> masked_string
    forward_map = {}       # masked_string -> original_string

    # Build left-to-right, substituting as we go (spans are
    # non-overlapping and sorted, so this is safe and simpler than
    # right-to-left splicing).
    out = []
    cursor = 0
    for start, end, label in spans:
        out.append(text[cursor:start])
        original = text[start:end]

        if label == "AGE":
            masked = "90+"
            # Deliberately NOT added to forward_map: HIPAA Safe Harbor
            # 3b requires generalizing, not hiding, so there is nothing
            # to rehydrate back to - "90+" is the correct final value.
        elif label == "DATE":
            shifted = shift_date(original, date_shift_days)
            if shifted is not None:
                masked = shifted
                forward_map[masked] = original
            else:
                key = (label, original.lower())
                if key not in memo:
                    counters[label] = counters.get(label, 0) + 1
                    memo[key] = f"[{label}_{counters[label]}]"
                masked = memo[key]
                forward_map[masked] = original
        else:
            key = (label, original.lower())
            if key not in memo:
                counters[label] = counters.get(label, 0) + 1
                memo[key] = f"[{label}_{counters[label]}]"
            masked = memo[key]
            forward_map[masked] = original

        out.append(masked)
        cursor = end
    out.append(text[cursor:])
    masked_text = "".join(out)

    selfcheck_leaks = []
    if run_selfcheck:
        raw_hits = _leak_selfcheck(masked_text)
        # A shifted date is DELIBERATELY left in DATE format (that's the
        # whole point - see module docstring), so regex will legitimately
        # re-match it here. That is not a leak: it's already a key in
        # forward_map, meaning rehydrate() can already account for it.
        # Only flag hits that are NOT something we already produced.
        selfcheck_leaks = [
            (s, e, label) for s, e, label in raw_hits
            if masked_text[s:e] not in forward_map
        ]
        if selfcheck_leaks:
            # Belt-and-suspenders: mask whatever the self-check found
            # too, right-to-left so offsets stay valid.
            chars = list(masked_text)
            for s, e, label in sorted(selfcheck_leaks, key=lambda x: -x[0]):
                original = masked_text[s:e]
                key = (label, original.lower())
                if key not in memo:
                    counters[label] = counters.get(label, 0) + 1
                    memo[key] = f"[{label}_{counters[label]}]"
                token = memo[key]
                forward_map[token] = original
                chars[s:e] = list(token)
            masked_text = "".join(chars)

    mapping = {
        "map": forward_map,
        "date_shift_days": date_shift_days,
        "spans_found": spans,
        "selfcheck_leaks": selfcheck_leaks,
    }
    return masked_text, mapping


_BRACKET_TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")


def rehydrate(response_text: str, mapping: dict):
    """Substitutes every masked_string in mapping["map"] back to its
    original value. Only ever replaces strings this exact mapping
    produced - see "REHYDRATION SECURITY" in the module docstring for
    what happens when the LLM echoes something that was never masked.

    Returns (rehydrated_text, warnings) where warnings lists any
    bracketed [LABEL_n]-shaped token found in the response that does
    NOT correspond to a key in this mapping - i.e. something the LLM
    introduced itself rather than something we masked.
    """
    forward_map = mapping["map"]
    text = response_text

    # Longest keys first, so e.g. "[NAME_10]" isn't corrupted by a
    # naive replace of "[NAME_1]" as a prefix.
    for masked in sorted(forward_map, key=len, reverse=True):
        text = text.replace(masked, forward_map[masked])

    warnings = []
    for m in _BRACKET_TOKEN_RE.finditer(response_text):
        if m.group(0) not in forward_map:
            warnings.append(m.group(0))

    return text, warnings


# --------------------------------------------------------------------- #
# Foundation LLM call
# --------------------------------------------------------------------- #
def _call_anthropic(prompt, model, api_key):
    import urllib.request
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    body = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            # Some providers sit behind Cloudflare, which blocks bare
            # urllib requests with no User-Agent (403, CF error 1010).
            "User-Agent": "phi-deid-gateway/1.0",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return "".join(block.get("text", "") for block in data.get("content", []))


def _call_openai_compatible(prompt, model, api_key, base_url, default_base_url,
                            default_model, api_key_env, key_optional=False):
    """Works with anything that speaks the OpenAI chat/completions shape.
    Shared by the 'groq' and 'openai_compatible' (Ollama, etc.) providers
    below - they differ only in their defaults."""
    import urllib.request
    base_url = (base_url or os.environ.get(f"{api_key_env}_BASE_URL")
                or default_base_url).rstrip("/")
    model = model or os.environ.get(f"{api_key_env}_MODEL", default_model)
    api_key = api_key or os.environ.get(api_key_env)
    if not api_key and not key_optional:
        raise RuntimeError(f"No API key found. Set ${api_key_env} or pass --api_key.")
    api_key = api_key or "not-needed"  # e.g. local Ollama, which ignores it

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Groq sits behind Cloudflare, which blocks bare urllib
            # requests with no User-Agent (403, CF error 1010) - this
            # header is the actual fix for that failure mode.
            "User-Agent": "phi-deid-gateway/1.0",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def call_foundation_llm(prompt: str, provider: str = "groq", model: str = None,
                        api_key: str = None, base_url: str = None,
                        dry_run: bool = False):
    """Sends masked_text (never raw text) to the foundation LLM and
    returns its raw text response.

    provider="groq"             -> Groq's OpenAI-compatible endpoint
                                    (https://api.groq.com/openai/v1),
                                    serving open-weight models (Llama,
                                    gpt-oss, Qwen) at very low latency.
                                    Reads $GROQ_API_KEY by default.
    provider="anthropic"        -> Claude via the Anthropic API.
    provider="openai_compatible" -> any other OpenAI-compatible
                                    chat/completions endpoint - most
                                    commonly a LOCAL Ollama server
                                    (default http://localhost:11434/v1,
                                    no API key, nothing leaves the
                                    machine), but also Together,
                                    OpenRouter, etc. via --base_url.

    Set dry_run=True (or omit credentials/a reachable endpoint) to
    exercise the rest of the pipeline without a real call - useful for
    testing deidentify/rehydrate offline.
    """
    if dry_run:
        return ("[DRY RUN - no LLM called] Echoing masked input back so "
                 "rehydrate() has something to demonstrate on:\n\n" + prompt)

    try:
        if provider == "anthropic":
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("[gateway.py] No ANTHROPIC_API_KEY set - falling "
                      "back to --dry_run behavior.")
                return call_foundation_llm(prompt, dry_run=True)
            return _call_anthropic(prompt, model, api_key)

        elif provider == "groq":
            # Groq has been deprecating llama-3.3-70b-versatile /
            # llama-3.1-8b-instant in favor of the gpt-oss family - if
            # you hit a "model decommissioned" error, check
            # console.groq.com/docs/models for the current list and
            # pass --llm_model to override.
            return _call_openai_compatible(
                prompt, model, api_key, base_url,
                default_base_url="https://api.groq.com/openai/v1",
                default_model="openai/gpt-oss-20b",
                api_key_env="GROQ_API_KEY",
            )

        elif provider == "openai_compatible":
            return _call_openai_compatible(
                prompt, model, api_key, base_url,
                default_base_url="http://localhost:11434/v1",
                default_model="llama3.2",
                api_key_env="OPENAI_API_KEY",
                key_optional=True,  # local Ollama doesn't need one
            )

        else:
            raise ValueError(f"Unknown provider: {provider!r} "
                              f"(expected 'groq', 'anthropic', or "
                              f"'openai_compatible')")
    except Exception as e:
        hint = ""
        if "403" in str(e):
            hint = (" A 403 here is usually one of: (1) missing "
                    "User-Agent header - should be fixed now; (2) the "
                    "model is blocked by your org's model permissions at "
                    "console.groq.com/settings/limits - try a different "
                    "--llm_model; (3) an invalid/expired key - regenerate "
                    "at console.groq.com/keys.")
        print(f"[gateway.py] LLM call failed ({provider}): {e}.{hint}\n"
              f"  Falling back to --dry_run behavior so the rest of the "
              f"pipeline still runs. If using Ollama, make sure it's "
              f"running: `ollama serve` and `ollama pull <model>` first.")
        return call_foundation_llm(prompt, dry_run=True)


# --------------------------------------------------------------------- #
# Full round trip - what P2.9's acceptance test actually exercises
# --------------------------------------------------------------------- #
def round_trip(raw_text: str, question: str, detect_fn, provider: str = "groq",
               model: str = None, api_key: str = None, base_url: str = None,
               dry_run: bool = False, verbose: bool = True):
    t0 = time.perf_counter()
    masked_text, mapping = deidentify(raw_text, detect_fn)
    t1 = time.perf_counter()

    if verbose:
        print("=" * 78)
        print("RAW TEXT (never leaves this process):")
        print(raw_text)
        print("-" * 78)
        print(f"MASKED TEXT (this is what the foundation LLM sees; "
              f"date_shift_days={mapping['date_shift_days']}):")
        print(masked_text)
        if mapping["selfcheck_leaks"]:
            print(f"[gateway.py] pre-send self-check caught and masked "
                  f"{len(mapping['selfcheck_leaks'])} additional span(s) "
                  f"the detector missed on the first pass.")
        print("-" * 78)

    prompt = f"{masked_text}\n\n---\n\n{question}"
    llm_response = call_foundation_llm(prompt, provider=provider, model=model,
                                       api_key=api_key, base_url=base_url,
                                       dry_run=dry_run)
    t2 = time.perf_counter()

    if verbose:
        print("FOUNDATION LLM RESPONSE (still masked at this point):")
        print(llm_response)
        print("-" * 78)

    rehydrated, warnings = rehydrate(llm_response, mapping)
    t3 = time.perf_counter()

    if verbose:
        print("REHYDRATED RESPONSE (returned to the caller):")
        print(rehydrated)
        if warnings:
            print(f"[gateway.py] WARNING: response contained {len(warnings)} "
                  f"bracket-shaped token(s) not in our mapping (possible "
                  f"LLM-introduced content, not a leak of real PHI): "
                  f"{warnings}")
        print(f"[gateway.py] timing: deidentify={t1-t0:.3f}s  "
              f"llm_call={t2-t1:.3f}s  rehydrate={t3-t2:.3f}s  "
              f"total={t3-t0:.3f}s")
        print("=" * 78)

    return {
        "masked_text": masked_text,
        "mapping": mapping,
        "llm_response": llm_response,
        "rehydrated_response": rehydrated,
        "rehydration_warnings": warnings,
        "timing_s": {
            "deidentify": t1 - t0,
            "llm_call": t2 - t1,
            "rehydrate": t3 - t2,
            "total": t3 - t0,
        },
    }


def _percentile(values, pct):
    """Simple linear-interpolation percentile - no numpy dependency
    needed for four to a few dozen samples."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def run_batch(examples, detect_fn, question, provider="groq", model=None,
             api_key=None, base_url=None, dry_run=False,
             report_path=None, verbose_each=False):
    """Runs round_trip() over a list of {"text", "source"} examples -
    by default WHITFIELD_GOLD_EXAMPLES, the real out-of-distribution
    excerpts from whitfield_gold.py. Produces the p50/p95 latency
    numbers P2.7 asks be reported, and optionally writes a full
    transcript to report_path for inclusion in your submission."""
    results = []
    print(f"\n[gateway.py] Running batch round-trip over {len(examples)} "
          f"example(s) (provider={provider}, dry_run={dry_run})...\n")

    for i, ex in enumerate(examples):
        source = ex.get("source", f"example_{i}")
        print(f"[{i+1}/{len(examples)}] {source} ...", end=" ", flush=True)
        res = round_trip(ex["text"], question, detect_fn, provider=provider,
                         model=model, api_key=api_key, base_url=base_url,
                         dry_run=dry_run, verbose=verbose_each)
        res["source"] = source
        results.append(res)
        t = res["timing_s"]
        flag = " [WARNINGS]" if res["rehydration_warnings"] else ""
        print(f"total={t['total']:.3f}s (llm={t['llm_call']:.3f}s){flag}")

    totals = [r["timing_s"]["total"] for r in results]
    llm_times = [r["timing_s"]["llm_call"] for r in results]
    deid_times = [r["timing_s"]["deidentify"] for r in results]
    n_warned = sum(1 for r in results if r["rehydration_warnings"])

    print("\n" + "=" * 78)
    print(f"BATCH SUMMARY ({len(results)} documents)")
    print("=" * 78)
    print(f"{'stage':<12} {'p50':>8} {'p95':>8} {'mean':>8} {'max':>8}")
    for label, vals in [("total", totals), ("llm_call", llm_times),
                        ("deidentify", deid_times)]:
        print(f"{label:<12} {_percentile(vals,50):>7.3f}s "
              f"{_percentile(vals,95):>7.3f}s {statistics.mean(vals):>7.3f}s "
              f"{max(vals):>7.3f}s")
    print(f"rehydration warnings: {n_warned}/{len(results)} document(s)")
    print("=" * 78)

    if report_path:
        _write_batch_report(report_path, results, totals, llm_times, deid_times)
        print(f"[gateway.py] Full transcript + latency report written to "
              f"{report_path}")

    return results


def _write_batch_report(path, results, totals, llm_times, deid_times):
    lines = ["# PHI Gateway - Round-Trip Batch Report", ""]
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Documents: {len(results)}")
    lines.append("")
    lines.append("## Latency (P2.7 - p50 / p95)")
    lines.append("")
    lines.append("| stage | p50 | p95 | mean | max |")
    lines.append("|---|---|---|---|---|")
    for label, vals in [("total", totals), ("llm_call", llm_times),
                        ("deidentify", deid_times)]:
        lines.append(f"| {label} | {_percentile(vals,50):.3f}s | "
                     f"{_percentile(vals,95):.3f}s | "
                     f"{statistics.mean(vals):.3f}s | {max(vals):.3f}s |")
    lines.append("")
    n_warned = sum(1 for r in results if r["rehydration_warnings"])
    lines.append(f"Rehydration warnings: {n_warned}/{len(results)} document(s)")
    lines.append("")
    lines.append("## Per-document transcripts")
    for r in results:
        lines.append("")
        lines.append(f"### {r['source']}")
        lines.append(f"timing: total={r['timing_s']['total']:.3f}s, "
                     f"llm_call={r['timing_s']['llm_call']:.3f}s, "
                     f"date_shift_days={r['mapping']['date_shift_days']}")
        if r["mapping"]["selfcheck_leaks"]:
            lines.append(f"self-check caught {len(r['mapping']['selfcheck_leaks'])} "
                         f"additional span(s) before send")
        lines.append("")
        lines.append("**Masked text sent to foundation LLM:**")
        lines.append("```")
        lines.append(r["masked_text"])
        lines.append("```")
        lines.append("**Foundation LLM response (masked):**")
        lines.append("```")
        lines.append(r["llm_response"])
        lines.append("```")
        lines.append("**Rehydrated response:**")
        lines.append("```")
        lines.append(r["rehydrated_response"])
        lines.append("```")
        if r["rehydration_warnings"]:
            lines.append(f"**WARNING - unrecognized bracket token(s) in "
                         f"response:** {r['rehydration_warnings']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------- #
_SAMPLE_TEXT = (
    "Patient Whitfield, Marcus D., DOB 03/14/1987, was seen 8 days after "
    "a rear-end motor vehicle collision on 02/11/2024. MRN PCG-4471902. "
    "Contact number (555) 219-4471. Attending S. Nakamura, MD, FACEP."
)
_SAMPLE_QUESTION = (
    "Summarize this note in two sentences for a referring physician, "
    "keeping the patient identifier placeholders exactly as written."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./phi_deid_model")
    parser.add_argument("--input", help="Path to a text file to de-identify. "
                                        "Omitted -> uses a built-in sample.")
    parser.add_argument("--question", default=_SAMPLE_QUESTION)
    parser.add_argument("--provider", default="groq",
                         choices=["groq", "anthropic", "openai_compatible"],
                         help="'groq' (default) for Groq's OpenAI-compatible "
                              "endpoint serving open-weight models - reads "
                              "$GROQ_API_KEY. 'anthropic' for Claude via the "
                              "Anthropic API. 'openai_compatible' for a "
                              "local Ollama server or any other endpoint "
                              "that speaks the OpenAI chat/completions "
                              "shape (Together, OpenRouter, ...).")
    parser.add_argument("--llm_model", default=None,
                         help="Foundation LLM model string, e.g. "
                              "'openai/gpt-oss-20b' (groq), "
                              "'claude-sonnet-4-5' (anthropic), or "
                              "'llama3.2' (openai_compatible/Ollama). "
                              "Check console.groq.com/docs/models for "
                              "Groq's current model list before relying "
                              "on the default - Groq deprecates model IDs "
                              "periodically.")
    parser.add_argument("--base_url", default=None,
                         help="Only used with --provider openai_compatible. "
                              "Defaults to http://localhost:11434/v1 "
                              "(Ollama). Point this at Groq/Together/"
                              "OpenRouter's endpoint to use a hosted open "
                              "model instead.")
    parser.add_argument("--api_key", default=None,
                         help="Overrides $GROQ_API_KEY / $ANTHROPIC_API_KEY / "
                              "$OPENAI_API_KEY (whichever matches --provider). "
                              "Not needed for a local Ollama server.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Skip the real LLM call (e.g. no internet / "
                              "no API key / Ollama not running) and just "
                              "exercise deidentify -> rehydrate on an "
                              "echoed prompt.")
    parser.add_argument("--skip_model", action="store_true",
                         help="Use regex-only detection (no fine-tuned "
                              "model loaded) - useful for a quick smoke "
                              "test without touching the GPU.")
    parser.add_argument("--batch", action="store_true",
                         help="Run the round trip over all documents in "
                              "WHITFIELD_GOLD_EXAMPLES (whitfield_gold.py) "
                              "instead of a single --input/sample text. "
                              "Prints p50/p95 latency (P2.7) across the "
                              "real out-of-distribution excerpts. "
                              "Overrides --input.")
    parser.add_argument("--report_out", default=None,
                         help="With --batch, write a full transcript + "
                              "latency report to this markdown file, e.g. "
                              "--report_out gateway_report.md. Handy "
                              "evidence to attach to your submission.")
    parser.add_argument("--verbose_each", action="store_true",
                         help="With --batch, also print the full raw/"
                              "masked/rehydrated text for every document "
                              "as it runs, not just the summary line.")
    args = parser.parse_args()

    if args.skip_model:
        detect_fn = build_ensemble_detector(model_detect_fn=None)
    else:
        try:
            detect_fn = build_ensemble_detector(model_dir=args.model_dir)
        except OSError:
            print(f"[gateway.py] No trained model found at {args.model_dir} - "
                  f"falling back to regex-only detection. Run train.py first, "
                  f"or pass --skip_model to silence this.")
            detect_fn = build_ensemble_detector(model_detect_fn=None)

    if args.batch:
        run_batch(WHITFIELD_GOLD_EXAMPLES, detect_fn, args.question,
                  provider=args.provider, model=args.llm_model,
                  api_key=args.api_key, base_url=args.base_url,
                  dry_run=args.dry_run, report_path=args.report_out,
                  verbose_each=args.verbose_each)
        return

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            raw_text = f.read()
    else:
        raw_text = _SAMPLE_TEXT

    round_trip(raw_text, args.question, detect_fn, provider=args.provider,
               model=args.llm_model, api_key=args.api_key,
               base_url=args.base_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main()