"""
labels.py
=========
Defines the entity label set used by the de-identification model and maps
each label back to its HIPAA Safe Harbor identifier category (P2.4 of the
brief). This module is the single source of truth for "what we cover".

We use a BIO tagging scheme: every label L has a B-L (beginning of span)
and I-L (inside span) tag, plus the "O" (outside) tag.
"""

# ---------------------------------------------------------------------------
# Entity types we detect in free text, and WHY each one maps to a HIPAA
# Safe Harbor category. Categories that cannot occur in *free clinical text*
# (they require pixel data or biometric signal capture) are explicitly
# excluded and documented below rather than silently dropped.
# ---------------------------------------------------------------------------

ENTITY_TO_HIPAA_CATEGORY = {
    "NAME":       "1  - Names (patients, relatives, providers, employers)",
    "LOCATION":   "2  - Geographic subdivisions smaller than a state "
                  "(street address, city, ZIP, facility name tied to a place)",
    "DATE":       "3  - All date elements except year (DOB, admission, "
                  "discharge, procedure dates)",
    "AGE":        "3b - Ages over 89 (special HIPAA edge case, must be "
                  "generalized to '90+' rather than a specific number)",
    "PHONE":      "4  - Telephone numbers",
    "FAX":        "5  - Fax numbers",
    "EMAIL":      "6  - Email addresses",
    "SSN":        "7  - Social Security numbers",
    "MRN":        "8  - Medical record numbers",
    "HEALTHPLAN": "9  - Health plan beneficiary numbers",
    "ACCOUNT":    "10 - Account numbers (billing / financial)",
    "LICENSE":    "11 - Certificate / license numbers (provider NPI, "
                  "state license, DEA number)",
    "VEHICLE":    "12 - Vehicle identifiers and license plate numbers",
    "DEVICE":     "13 - Device identifiers and serial numbers",
    "URL":        "14 - URLs",
    "IP":         "15 - IP addresses",
    "ID_OTHER":   "18 - Any other unique identifying number, characteristic "
                  "or code (claim numbers, employee IDs, run numbers, "
                  "accession numbers, etc.)",
}

# Categories explicitly OUT of scope for this text-only gateway, and why.
EXCLUDED_HIPAA_CATEGORIES = {
    16: "Biometric identifiers (fingerprints, voiceprints) - not "
        "representable in free text; would require signal/waveform input.",
    17: "Full-face photographs and comparable images - this gateway "
        "processes text only. An image gateway (face detection + blurring) "
        "is a separate service and out of scope for P2.",
}

LABELS = sorted(ENTITY_TO_HIPAA_CATEGORY.keys())

# Build the BIO tag list: O, B-NAME, I-NAME, B-LOCATION, I-LOCATION, ...
BIO_TAGS = ["O"]
for lbl in LABELS:
    BIO_TAGS.append(f"B-{lbl}")
    BIO_TAGS.append(f"I-{lbl}")

TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}

# Recall is what matters clinically: a missed identifier is a data breach,
# a false positive is (at worst) an over-redacted token. We use this table
# to weight per-class loss during training (see train.py) so the model is
# penalized more heavily for missing high-risk categories such as SSN/MRN.
RECALL_CRITICAL_WEIGHT = {
    "SSN": 5.0, "MRN": 4.0, "HEALTHPLAN": 4.0, "ACCOUNT": 3.0,
    "NAME": 3.0, "PHONE": 2.5, "EMAIL": 2.5, "DEVICE": 2.0,
    "LICENSE": 2.0, "VEHICLE": 2.0, "IP": 2.0, "URL": 1.5,
    "LOCATION": 2.0, "DATE": 2.0, "AGE": 2.0, "FAX": 2.0, "ID_OTHER": 2.5,
}