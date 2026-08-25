"""
whitfield_gold.py
==================
Hand-labeled excerpts taken VERBATIM from
Synthetic_Medical_Record_Exercise_Whitfield_1.pdf (itself synthetic /
fictional per that document's own header - no real patient data at any
point in this pipeline).

Why this file exists: everything in data_gen.py is still synthetic
template output. Even with disjoint train/eval template pools, the eval
set is drawn from generators I wrote myself, which share vocabulary and
structural DNA with the training generators in ways I'm not fully aware
of. The Whitfield document is real prose written independently of this
codebase - different sentence rhythm, real clinical phrasing, genuinely
unseen surface forms. This is the "single most informative number" in
the same sense P3.6 describes for the FHIR pipeline: recovery on data the
system was never shaped around.

Entities are specified as (substring, label) pairs and converted to
character offsets automatically via `locate()`, rather than hand-counted
offsets, because hand-counting offsets in a paragraph is exactly the kind
of manual step that silently introduces off-by-one labeling errors.
"""
import re


def locate(text: str, tagged_substrings: list[tuple[str, str]]):
    """Convert a list of (exact_substring, label) pairs into
    (start, end, label) character spans, searching left-to-right so
    repeated identical substrings (e.g. the same date appearing twice)
    are matched to successive occurrences in order rather than all
    collapsing onto the first hit."""
    entities = []
    cursor = 0
    for substr, label in tagged_substrings:
        idx = text.find(substr, cursor)
        if idx == -1:
            # fall back to searching from the start, in case ordering in
            # the list doesn't match left-to-right order in the text
            idx = text.find(substr)
            if idx == -1:
                raise ValueError(f"Could not locate substring {substr!r} in "
                                  f"gold text - check for a transcription typo.")
        entities.append((idx, idx + len(substr), label))
        cursor = idx + len(substr)
    entities.sort(key=lambda x: x[0])
    return entities


# ---------------------------------------------------------------------- #
# Excerpt 1: EMS run report header + narrative (LTM-PI-00001)
# Real prose, mixed casing, dense identifier clustering, plus DELIBERATE
# non-PHI numeric content (vitals, ESI/GCS scores) as negative controls.
# ---------------------------------------------------------------------- #
EX1_TEXT = (
    "Patient Name Whitfield, Marcus D.\n"
    "Date of Birth 03/14/1987 (Age 36)\n"
    "Incident Date / Time 02/11/2024 17:42\n"
    "Receiving Facility Piedmont County General Hospital, ED\n"
    "Run Number EMS-2024-03318\n"
    "Patient is a 36-year-old male, alert and oriented x4, GCS 15. "
    "Complains of immediate onset posterior neck pain. "
    "17:56 BP 148/88, HR 96, RR 20, SpO2 98% RA, Pain 7/10\n"
    "Crew: J. Okafor, NRP (Lead) / T. Bianchi, EMT-B."
)
EX1_ENTITIES = locate(EX1_TEXT, [
    ("Whitfield, Marcus D.", "NAME"),
    ("03/14/1987", "DATE"),
    ("02/11/2024", "DATE"),
    ("Piedmont County General Hospital", "LOCATION"),
    ("EMS-2024-03318", "ID_OTHER"),
    ("J. Okafor", "NAME"),
    ("T. Bianchi", "NAME"),
    # NOTE: "Age 36", "GCS 15", "BP 148/88", "HR 96", "RR 20", "SpO2 98%",
    # "Pain 7/10" are intentionally NOT tagged - they are clinical vitals
    # / scores, not HIPAA identifiers, and are the negative-control check.
])

# ---------------------------------------------------------------------- #
# Excerpt 2: ED triage record - allergy + intake questionnaire style,
# includes an MRN/account-style ID and an all-caps facility header.
# ---------------------------------------------------------------------- #
EX2_TEXT = (
    "PIEDMONT COUNTY GENERAL HOSPITAL\n"
    "Patient Whitfield, Marcus D.\n"
    "MRN PCG-4471902\n"
    "Account / Encounter ED-24-0211-556\n"
    "Arrival 02/11/2024 18:19 via EMS\n"
    "Allergies CODEINE (rash, documented by patient) ; NKA otherwise\n"
    "Currently working? YES - warehouse logistics coordinator, full duty\n"
    "Electronically signed: R. Delacroix, RN | 02/11/2024 21:58"
)
EX2_ENTITIES = locate(EX2_TEXT, [
    ("PIEDMONT COUNTY GENERAL HOSPITAL", "LOCATION"),
    ("Whitfield, Marcus D.", "NAME"),
    ("PCG-4471902", "MRN"),
    ("ED-24-0211-556", "ACCOUNT"),
    ("02/11/2024", "DATE"),
    ("R. Delacroix", "NAME"),
    ("02/11/2024", "DATE"),
    # NOTE: "CODEINE" (an allergy, clinical fact not identifier) and
    # "warehouse logistics coordinator" (occupation, not itself a Safe
    # Harbor category) are intentionally NOT tagged.
])

# ---------------------------------------------------------------------- #
# Excerpt 3: physician note prose - the genuinely hard case, since real
# clinical narrative embeds names mid-sentence with no anchor keyword
# ("SSN:", "MRN:") the way structured fields do.
# ---------------------------------------------------------------------- #
EX3_TEXT = (
    "The patient is a 36-year-old male with no significant past medical "
    "history who presents by EMS following a motor vehicle collision. "
    "Attending S. Nakamura, MD, FACEP evaluated the patient. Referral to "
    "orthopedic spine surgery, Dr. Halloway, was placed the same day. "
    "Straight leg raise positive on the right at approximately 45 degrees, "
    "negative on the left. Electronically signed: Sara Nakamura, MD | "
    "02/11/2024 22:31"
)
EX3_ENTITIES = locate(EX3_TEXT, [
    ("S. Nakamura", "NAME"),
    ("Dr. Halloway", "NAME"),
    ("Sara Nakamura", "NAME"),
    ("02/11/2024", "DATE"),
    # NOTE: "45 degrees" is an exam finding, not PHI - negative control.
])

# ---------------------------------------------------------------------- #
# Excerpt 4: pharmacy dispensing record - dense structured IDs, a URL-free
# but phone/insurance-style block, tests recall on rapid-fire identifiers.
# ---------------------------------------------------------------------- #
EX4_TEXT = (
    "GREENLEAF COMMUNITY PHARMACY #218\n"
    "Patient Whitfield, Marcus D.\n"
    "Insurance Meridian Health Plan, Member 88-2210-447\n"
    "02/11/2024 Naproxen 500 mg tab, qty 30, Rx by Nakamura S MD\n"
    "09/10/2024 Oxycodone/Acetaminophen 5-325 mg tab, qty 20, Rx by Halloway G MD"
)
EX4_ENTITIES = locate(EX4_TEXT, [
    ("GREENLEAF COMMUNITY PHARMACY", "LOCATION"),
    ("Whitfield, Marcus D.", "NAME"),
    ("88-2210-447", "HEALTHPLAN"),
    ("02/11/2024", "DATE"),
    ("Nakamura S", "NAME"),
    ("09/10/2024", "DATE"),
    ("Halloway G", "NAME"),
])

WHITFIELD_GOLD_EXAMPLES = [
    {"text": EX1_TEXT, "entities": EX1_ENTITIES, "source": "EMS run report"},
    {"text": EX2_TEXT, "entities": EX2_ENTITIES, "source": "ED triage record"},
    {"text": EX3_TEXT, "entities": EX3_ENTITIES, "source": "Physician note"},
    {"text": EX4_TEXT, "entities": EX4_ENTITIES, "source": "Pharmacy record"},
]

if __name__ == "__main__":
    for ex in WHITFIELD_GOLD_EXAMPLES:
        print(f"--- {ex['source']} ---")
        for s, e, lbl in ex["entities"]:
            print(f"  {lbl:12s} {ex['text'][s:e]!r}")