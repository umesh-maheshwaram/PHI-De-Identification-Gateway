"""
regex_baseline.py
==================
Baseline (a) from P2.7: regex-only identifier detection.

Regex is genuinely the *right* tool for the structured, high-entropy
categories (SSN, phone, email, URL, IP, dates in numeric format) - it gets
near-perfect recall on these because they have a rigid grammar. It is
included in the production gateway itself (see gateway.py), not just as a
throwaway baseline, because a trained NER model is comparatively worse at
memorizing "3 digits - 2 digits - 4 digits" than a regex is.

Where regex fails, and why we still need the trained model (P2.5):
  - NAME: "Dr. Parkinson diagnosed Parkinson's" - no surface pattern
    distinguishes the two occurrences.
  - LOCATION / facility names embedded in prose.
  - ID_OTHER: free-form identifiers with no fixed grammar.
  - PHONE vs FAX: identical digit grammar ("nnn-nnn-nnnn"). A pure regex
    genuinely cannot tell these apart by pattern alone - the only signal
    is the nearby word "fax" vs "phone"/"call"/"contact". We approximate
    this with a context window (see FAX_CONTEXT_WINDOW below), which is
    itself a small rule-based hack layered on top of regex, not a clean
    grammar match - documenting this honestly rather than papering over
    it, per the brief's instruction to report limitations plainly.
"""
import re

# Each pattern returns (start, end, label). Ordered so more specific
# patterns (SSN) are tried before more general ones that could overlap.
PATTERNS = [
    ("SSN",     re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL",   re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("URL",     re.compile(r"\bhttps?://[^\s,)]+")),
    ("IP",      re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("PHONE_OR_FAX", re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b")),
    ("MRN",     re.compile(r"\b(?:MRN|PCG|MER)-?\d{5,8}\b")),
    ("HEALTHPLAN", re.compile(r"\b\d{2}-\d{4}-\d{3}\b")),
    ("ACCOUNT", re.compile(r"\bACCT-\d{6,8}\b|\b(?:ED|OR|PT|RAD)-\d{2}-\d{4}-\d{3}\b")),
    ("LICENSE", re.compile(r"\bNPI-\d{9,10}\b")),
    ("DEVICE",  re.compile(r"\bSN\d{5,7}-[A-Z]\b")),
    ("VEHICLE", re.compile(r"\b[A-Z]{3}-\d{4}\b")),
    ("DATE",    re.compile(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
        r"|\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},\s*\d{4}\b")),
    ("AGE",     re.compile(r"\bage\s+(9[0-9]|1\d{2})\b", re.IGNORECASE)),
    ("ID_OTHER", re.compile(r"\b(?:EMS|CLAIM|PI)-\d{4}-\d{3,5}\b", re.IGNORECASE)),
]

FAX_CONTEXT_WINDOW = 25  # chars of preceding text scanned for the word "fax"
FAX_CONTEXT_RE = re.compile(r"\bfax\b", re.IGNORECASE)


def regex_detect(text: str):
    """Return a list of (start, end, label) spans found by regex alone.

    Overlaps are resolved by keeping the first (longest-priority) match and
    discarding subsequent matches whose span intersects an already-claimed
    region, so e.g. a DATE inside an already-matched pattern isn't double
    counted.
    """
    claimed = [False] * (len(text) + 1)
    spans = []
    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.span()
            resolved_label = label  # never mutate the outer loop var `label` -
            # doing so previously caused every match AFTER the first
            # PHONE_OR_FAX in a document to silently inherit the prior
            # match's resolved label instead of being re-evaluated, since
            # Python for-loops don't create a new binding per iteration.
            if label == "AGE":
                # only the number itself is the identifier, not "age "
                num_match = re.search(r"\d+", m.group())
                s = m.start() + num_match.start()
                e = m.start() + num_match.end()
            if label == "PHONE_OR_FAX":
                # disambiguate via nearby context - see module docstring
                window_start = max(0, s - FAX_CONTEXT_WINDOW)
                context = text[window_start:s]
                resolved_label = "FAX" if FAX_CONTEXT_RE.search(context) else "PHONE"
            if any(claimed[s:e]):
                continue
            for i in range(s, e):
                claimed[i] = True
            spans.append((s, e, resolved_label))
    spans.sort(key=lambda x: x[0])
    return spans


if __name__ == "__main__":
    sample = ("Patient Marcus Whitfield, SSN 471-88-2091, DOB 03/14/1987, "
               "reached at (555) 219-4471 or marcus.w@example.com. "
               "Fax records to 555-219-4472.")
    for s, e, lbl in regex_detect(sample):
        print(lbl, repr(sample[s:e]))