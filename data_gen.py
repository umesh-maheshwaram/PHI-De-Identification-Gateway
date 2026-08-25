"""
data_gen.py (v2)
=================
Rebuilt after the v1 model hit precision/recall/F1 = 1.0 from epoch 1
onward - a red flag, not a good result. Root cause: only 8 templates,
train and eval drawn from the SAME templates, so the model memorized
surface position ("the token right after 'SSN'") instead of learning to
recognize identifiers in context. A perfectly-scoring model that has
memorized 8 sentence shapes tells us nothing about the real task.

Three structural fixes, in the order we diagnosed them:

  1. TRAIN_TEMPLATES and EVAL_TEMPLATES are now DISJOINT template pools.
     generate_dataset(split="train") only draws from one set,
     generate_dataset(split="eval") only from the other. This is the
     minimum bar for "eval measures generalization, not memorization."

  2. Deliberate ambiguity cases are now first-class citizens, directly
     targeting P2.5's "Dr. Parkinson diagnosed Parkinson's" example:
       - EPONYM_DISEASES: surnames that are also disease names
         (Parkinson, Alzheimer, Crohn, Hodgkin, Graves, Addison, Wilson,
         Cushing, Raynaud, Bell). Templates place BOTH the physician
         mention (tag as NAME) and the disease mention (tag as O, not
         PHI) in the same or adjacent sentences.
       - Facility names built from a surname + institute suffix
         ("Halloway Orthopedic Spine Institute") are tagged as a single
         LOCATION span, while a bare "Dr. Halloway" a sentence later is
         tagged NAME. Same surname, two different correct labels
         depending on context - exactly what a memorized position
         heuristic cannot solve.

  3. Explicit negative controls for numeric near-misses, so the model
     learns identifier GRAMMAR, not "any digit-heavy span":
       - vital signs / lab values (BP 128/80, pain 7/10) -> O
       - relative time references ("3 days post-op", "day 5") -> O
       - ages UNDER 90 -> O (only ages >89 are a HIPAA identifier -
         P2.4 item 3b; the model must learn the threshold, not just
         "number near the word age")
       - ICD-10 / CPT codes -> O (these identify a condition/procedure,
         not a person, and are explicitly not a Safe Harbor category)

Casing is deliberately mixed (ALL-CAPS headers, Title Case prose, and
lowercase) within both pools, since the real Whitfield-style documents do
this too, and the base model (distilbert-base-UNCASED) needs training
signal that doesn't rely on casing as a shortcut.
"""
import random
import re

try:
    from faker import Faker
    _fake = Faker()
    HAVE_FAKER = True
except ImportError:
    HAVE_FAKER = False

# ---------------------------------------------------------------------- #
# Vocab
# ---------------------------------------------------------------------- #
FIRST_NAMES = ["Marcus", "Elena", "David", "Priya", "Samuel", "Grace",
               "Wei", "Fatima", "Connor", "Ingrid", "Malik", "Rosa",
               "Theo", "Naomi", "Julian", "Anika", "Bruno", "Sofia",
               "Lydia", "Gregory", "Karine", "Darius", "Helena", "Rita"]
LAST_NAMES = ["Whitfield", "Okafor", "Bianchi", "Delacroix", "Nakamura",
              "Osei-Bonsu", "Petrosyan", "Ramachandran", "Halloway",
              "Farhadi", "Vasquez-Lund", "Nkemdirim", "Sokolova", "Iverson",
              "Castellanos", "Mwangi", "Lindqvist", "Brant"]
CITIES = ["Piedmont", "Rosewood", "Fairview", "Kingston", "Millbrook",
          "Cedar Falls", "Northgate", "Salem", "Greenfield"]
STREETS = ["Wellstone Parkway", "Rosewood Ave", "Main Street",
           "Harbor Road", "5th Avenue", "Elm Street"]
FACILITY_SUFFIXES = ["Orthopedic Spine Institute", "Family Medicine Associates",
                      "Rehabilitation & Sports Therapy", "Interventional Pain & Spine",
                      "Imaging Partners", "General Hospital", "Urgent Care"]
SPECIALTIES = ["Emergency Medicine", "Orthopedic Spine Surgery", "Radiology",
               "Physical Therapy", "Family Medicine", "Neurology",
               "Interventional Pain Management", "Surgery"]
CREDENTIALS = ["MD", "DO", "PA-C", "RN", "PT, DPT", "MD, FACEP"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Surnames that double as disease/syndrome eponyms - the core hard case.
EPONYM_DISEASES = {
    "Parkinson": "Parkinson's disease", "Alzheimer": "Alzheimer's disease",
    "Crohn": "Crohn's disease", "Hodgkin": "Hodgkin's lymphoma",
    "Graves": "Graves' disease", "Addison": "Addison's disease",
    "Wilson": "Wilson's disease", "Cushing": "Cushing's syndrome",
    "Raynaud": "Raynaud's phenomenon", "Bell": "Bell's palsy",
}

ICD_CODES = ["S13.4XXA", "S33.5XXA", "M54.41", "V49.9XXA", "M51.26", "G56.00"]
CPT_CODES = ["99284", "72125", "72100", "97162", "63030", "64483"]


def _rand_name():
    if HAVE_FAKER:
        return _fake.name()
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _rand_name_reversed():
    """'Lastname, Firstname M.' - the running-header name format used on
    EVERY page of the real Whitfield document. Mining that document's
    actual structure surfaced this gap: none of the v1/v2 templates ever
    generated reversed name order, so the model had zero training signal
    for it and missed 'Whitfield, Marcus D.' on every occurrence during
    the real-document eval."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    middle_initial = random.choice("ABCDEFGHJKLMNPRSTW")
    return f"{last}, {first} {middle_initial}."


def _rand_name_terse():
    """'Surname Initial' with NO comma - a third, distinct name order
    mined from the real document's pharmacy dispensing record ('Rx by
    Nakamura S MD', 'Rx by Halloway G MD'). This was missed on the real-
    document eval even after fixing the comma-separated reversed form,
    because it's a genuinely different surface pattern - no comma, no
    period after the initial, often followed directly by a credential."""
    last = random.choice(LAST_NAMES)
    initial = random.choice("ABCDEFGHJKLMNPRSTW")
    return f"{last} {initial}"


def _rand_signature_line():
    """Models 'Electronically signed: {NAME}, MD | {DATE} {TIME}' -
    mined from the operative report, physician note, and PT discharge
    summary sections of the real document (none of which are used in
    whitfield_gold.py's eval excerpts)."""
    name = _rand_name()
    cred = random.choice(CREDENTIALS)
    date = _rand_date()
    time = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}"
    return name, cred, date, time


def _rand_surname():
    return random.choice(LAST_NAMES)


def _rand_date():
    if HAVE_FAKER:
        return _fake.date(pattern="%m/%d/%Y")
    return f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{random.randint(1960,2024)}"


def _rand_long_date():
    return f"{random.choice(MONTHS)} {random.randint(1,28)}, {random.randint(1960,2024)}"


def _rand_phone():
    return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"


def _rand_fax():
    """Returns the bare number only. A human labeler tags the digits as
    the identifier, not the surrounding word 'fax' - the template text
    supplies that context ('Fax records to {FAX}.'), the entity span
    should not. (v2 bug: this used to append literal ' (fax)' into the
    RETURNED VALUE, which then got tagged as part of the entity itself,
    producing garbled doubled text like 'by fax (555-1234 (fax))' when
    combined with a template that already said 'fax'.)"""
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"


def _rand_email(name=None):
    local = (name or _rand_name()).lower().replace(" ", ".").replace("'", "")
    domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "meridianhealth.org"])
    return f"{local}@{domain}"


def _rand_ssn():
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"


def _rand_mrn():
    return f"{random.choice(['PCG','MER','MRN'])}-{random.randint(100000,9999999)}"


def _rand_healthplan():
    return f"{random.randint(10,99)}-{random.randint(1000,9999)}-{random.randint(100,999)}"


def _rand_account():
    """Two real-world account/encounter ID grammars, both mined from the
    document - the department-prefixed encounter format ('ED-24-0211-556')
    was entirely absent before and was missed by both regex AND the
    trained model on every real-document run so far."""
    if random.random() < 0.5:
        return f"ACCT-{random.randint(1000000,9999999)}"
    dept = random.choice(["ED", "OR", "PT", "RAD"])
    yy = random.randint(20, 26)
    mmdd = f"{random.randint(1,12):02d}{random.randint(1,28):02d}"
    seq = random.randint(100, 999)
    return f"{dept}-{yy}-{mmdd}-{seq}"


def _rand_license():
    return f"NPI-{random.randint(1000000000,1999999999)}"


def _rand_vehicle():
    letters = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=3))
    digits = "".join(random.choices("0123456789", k=4))
    return f"{letters}-{digits}"


def _rand_device():
    return f"SN{random.randint(100000,999999)}-{random.choice(['A','B','C'])}"


def _rand_url():
    return f"https://patientportal.{random.choice(['meridianhealth','piedmontcgh','valleyimaging'])}.com/records/{random.randint(1000,9999)}"


def _rand_ip():
    return f"{random.randint(10,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _rand_age_over_89():
    return str(random.randint(90, 104))


def _rand_age_under_90():
    """Negative control: a number near the word 'age' that must NOT be
    tagged, because only ages over 89 are a Safe Harbor identifier."""
    return str(random.randint(1, 89))


def _rand_id_other():
    return f"EMS-{random.randint(2020,2025)}-{random.randint(1000,9999)}"


def _rand_address():
    return f"{random.randint(100,9999)} {random.choice(STREETS)}, {random.choice(CITIES)}"


def _rand_facility(surname=None):
    """Real facility names split roughly into two families that were NOT
    both represented before: person-named ('Halloway Orthopedic Spine
    Institute') and place-named ('Piedmont County General Hospital',
    'Greenleaf Community Pharmacy'). The real document's LOCATION misses
    ('PIEDMONT COUNTY GENERAL HOSPITAL', 'GREENLEAF COMMUNITY PHARMACY')
    were both place-named - a pattern this generator never produced at
    all until now."""
    style = random.random()
    if style < 0.4 and surname:
        return f"{surname} {random.choice(FACILITY_SUFFIXES)}"
    elif style < 0.7:
        return f"{random.choice(CITIES)} County General Hospital"
    else:
        pharmacy_num = random.randint(100, 999)
        return f"{random.choice(CITIES)} Community Pharmacy #{pharmacy_num}"


def _rand_relative_time():
    """Negative control: clinically meaningful but NOT a Safe Harbor date
    (no absolute calendar date is present)."""
    return random.choice([
        f"{random.randint(1,14)} days post-op",
        f"{random.randint(2,8)} weeks after the injury",
        f"day {random.randint(1,10)} of treatment",
        "three days ago", "earlier this evening", "at the eight-week follow-up",
    ])


def _rand_vitals():
    """Negative control: numeric clinical data, not an identifier."""
    return random.choice([
        f"BP {random.randint(100,160)}/{random.randint(60,100)}",
        f"pain {random.randint(1,10)}/10",
        f"HR {random.randint(55,110)}",
        f"SpO2 {random.randint(94,100)}%",
    ])


def _rand_code():
    return random.choice(ICD_CODES + CPT_CODES)


# ---------------------------------------------------------------------- #
# Field generators keyed by placeholder name used in templates.
# ---------------------------------------------------------------------- #
FIELD_GENERATORS = {
    "NAME": _rand_name,
    "SURNAME_NAME": lambda: f"Dr. {_rand_surname()}",  # for eponym pairing
    "DATE": lambda: random.choice([_rand_date(), _rand_long_date()]),
    "PHONE": _rand_phone,
    "FAX": _rand_fax,
    "EMAIL": lambda: _rand_email(),
    "SSN": _rand_ssn,
    "MRN": _rand_mrn,
    "HEALTHPLAN": _rand_healthplan,
    "ACCOUNT": _rand_account,
    "LICENSE": _rand_license,
    "VEHICLE": _rand_vehicle,
    "DEVICE": _rand_device,
    "URL": _rand_url,
    "IP": _rand_ip,
    "AGE": _rand_age_over_89,
    "ID_OTHER": _rand_id_other,
    "LOCATION": lambda: random.choice([_rand_address(), _rand_facility()]),
}

# Non-tagged placeholders (rendered into text, but NOT added to entities).
# Used for negative controls - see module docstring point 3.
NEGATIVE_GENERATORS = {
    "AGE_NEG": _rand_age_under_90,
    "RELTIME": _rand_relative_time,
    "VITALS": _rand_vitals,
    "CODE": _rand_code,
    "SPECIALTY": lambda: random.choice(SPECIALTIES),  # a department name is
    # not itself a Safe Harbor identifier - negative control
}

# ---------------------------------------------------------------------- #
# TRAIN_TEMPLATES and EVAL_TEMPLATES are DISJOINT. Never share a template
# string between the two lists - that discipline is the whole point.
# ---------------------------------------------------------------------- #
TRAIN_TEMPLATES = [
    "Patient {NAME}, DOB {DATE}, presents to {LOCATION} on {DATE} with "
    "worsening low back pain. Contact number {PHONE}, email {EMAIL}.",

    "{NAME} (SSN {SSN}, MRN {MRN}) was seen by Dr. {NAME} at {LOCATION}. "
    "Health plan beneficiary number {HEALTHPLAN}. Fax records to {FAX}.",

    "Ambulance run {ID_OTHER} transported {NAME}, age {AGE}, from the scene "
    "on {DATE} to {LOCATION}. Vehicle involved bore plate {VEHICLE}.",

    "Follow-up scheduled for {NAME} on {DATE}. Billing account {ACCOUNT}. "
    "Access records at {URL} (last login from {IP}).",

    "Device serial {DEVICE} was implanted during the procedure performed on "
    "{DATE} by Dr. {NAME}, license {LICENSE}, at {LOCATION}.",

    "Mr./Ms. {NAME} called from {PHONE} regarding results dated {DATE}. "
    "Please reply to {EMAIL} or fax {FAX}.",

    # --- eponym hard cases (train) ---
    "{SURNAME_NAME} evaluated the patient and documented a working "
    "diagnosis consistent with {EPONYM}, unrelated to the treating "
    "physician's own name.",

    "Consultation requested with {SURNAME_NAME} regarding possible "
    "{EPONYM}; the patient's neurological exam was otherwise nonfocal.",

    # --- facility-vs-person hard cases (train) ---
    "Records were transferred to {LOCATION_FACILITY} for continuity of "
    "care following discharge on {DATE}.",

    "Two weeks later {SURNAME_NAME} reviewed the imaging personally and "
    "recommended the patient return to {LOCATION_FACILITY} for follow-up.",

    # ------------------------------------------------------------------ #
    # MINED from real (non-eval) pages of Synthetic_Medical_Record_
    # Exercise_Whitfield_1.pdf, generalized (values re-randomized, never
    # the literal document text). Sources, all DISJOINT from the four
    # excerpts used in whitfield_gold.py (EMS report, ED triage, physician
    # note, pharmacy record):
    #   - Northgate Family Medicine office visit note (LTM-PI-00006)
    #   - PT initial evaluation (LTM-PI-00007)
    #   - MRI lumbar spine report (LTM-PI-00008)
    #   - Halloway orthopedic consultation (LTM-PI-00009)
    #   - Operative report (LTM-PI-00015)
    # These patterns repeat on nearly every page of the real document and
    # had ZERO representation in the original hand-written templates -
    # in particular, reversed 'Lastname, Firstname M.' name order, which
    # explains most of the false negatives on the real-document eval run.
    # ------------------------------------------------------------------ #

    # running page-header pattern: "LASTNAME, FIRSTNAME M. | DOB .. | CLAIM .."
    "{NAME_REV} | DOB {DATE} | CLAIM {ID_OTHER} PRODUCED {DATE}",

    # facility + department header block
    "{LOCATION}\nDepartment of {SPECIALTY}",

    # referring / attending provider one-liner
    "Referring Provider {SURNAME_NAME}, {DATE}",
    "Attending {SURNAME_NAME} evaluated the patient in the {SPECIALTY} "
    "department at {LOCATION}.",

    # MRN / account header block (structured key-value lines)
    "MRN {MRN}\nAccount / Encounter {ACCOUNT}\nArrival {DATE} via EMS",

    # signature-line pattern: "Electronically signed: Name, MD | date time"
    "Electronically signed: {NAME} | {DATE}",
    "Electronically signed: {NAME}, MD | {DATE} 22:31",

    # reversed name embedded mid-sentence (not just as a header)
    "Patient {NAME_REV} was evaluated at {LOCATION} on {DATE} for "
    "continued follow-up care.",

    # consultation / referral note mixing reversed and normal name order
    # in the SAME line - the genuinely hard mixed-order case
    "{NAME_REV} was referred to {SURNAME_NAME} for further evaluation "
    "at {LOCATION}.",

    # negative control mined from the same pages: relative post-op timing
    # phrased the way real operative/follow-up notes actually phrase it
    "Patient returns {RELTIME} following the procedure performed at "
    "{LOCATION}. No new identifying details in this sentence.",

    # --- negative controls (train) ---
    "Vitals at intake: {VITALS}. Patient reports symptom onset {RELTIME}. "
    "No identifying information in this line.",

    "Diagnosis coded as {CODE}. The patient is {AGE_NEG} years old and "
    "denies prior similar episodes.",

    "Reflexes 2+ and symmetric bilaterally. Sensation intact. No red flag "
    "features noted. Straight leg raise negative.",  # pure negative, no PHI

    "Motor exam nonfocal throughout. Gait steady. {VITALS}, recorded "
    "{RELTIME}.",  # pure negative with distractor numbers

    # ------------------------------------------------------------------ #
    # SECOND context each for SSN / VEHICLE / DEVICE / HEALTHPLAN, which
    # previously depended on exactly ONE template apiece (see the shared
    # single-template diagnosis). A model can't tell "learned the grammar"
    # from "memorized the one slot it always appears in" without at least
    # two independent phrasings per class.
    # ------------------------------------------------------------------ #
    "Identity confirmed via SSN {SSN} at check-in; no other identifying "
    "documents were presented.",

    "Insurance carrier lists the beneficiary under health plan number "
    "{HEALTHPLAN}, separate from the medical record number on file.",

    "The responding officer noted the second vehicle bore plate "
    "{VEHICLE} and had front-end damage.",

    "Intraoperative fluoroscopy confirmed correct placement of hardware; "
    "device serial {DEVICE} was logged in the implant registry.",

    # NAME_TERSE ('Surname Initial', no comma) - mined from the pharmacy
    # dispensing record pattern ('Rx by Nakamura S MD').
    "Rx by {NAME_TERSE} MD, filled at {LOCATION} on {DATE}.",
    "Prescribing provider on file: {NAME_TERSE}, credentialed at "
    "{LOCATION}.",
]

EVAL_TEMPLATES = [
    "On {DATE}, {NAME} was transported by EMS unit {ID_OTHER} to "
    "{LOCATION}; next of kin can be reached at {PHONE}.",

    "Insurance claim filed under account {ACCOUNT} for beneficiary "
    "{NAME}, health plan number {HEALTHPLAN}, treated at {LOCATION}.",

    "Please route correspondence for {NAME} (MRN {MRN}) to {EMAIL}; "
    "physician of record holds license {LICENSE}.",

    "{NAME}'s vehicle, plate {VEHICLE}, was struck on {DATE}. SSN on file: "
    "{SSN}. Implanted device serial {DEVICE} noted intraoperatively.",

    "A message was left for {NAME} at {PHONE}; portal access remains at "
    "{URL}, last accessed from IP {IP} on {DATE}.",

    "{NAME}, age {AGE}, was referred by fax ({FAX}) to Dr. {NAME} for "
    "further evaluation at {LOCATION}.",

    # --- eponym hard cases (eval - DIFFERENT phrasing than train) ---
    "The differential includes {EPONYM}, though {SURNAME_NAME} favors a "
    "structural rather than neurodegenerative cause.",

    "{SURNAME_NAME}'s note documents a prior discussion of {EPONYM} with "
    "the family, separate from the surgeon's own identity.",

    # --- facility-vs-person hard cases (eval - different phrasing) ---
    "Prior imaging obtained at {LOCATION_FACILITY} was reviewed before "
    "{SURNAME_NAME} proceeded with the consultation.",

    "The patient was later seen independently by {SURNAME_NAME}, whose "
    "clinic is unaffiliated with {LOCATION_FACILITY}.",

    # --- negative controls (eval) ---
    "Labs remarkable for a mild leukocytosis; patient afebrile. {VITALS}. "
    "No new complaints {RELTIME}.",

    "Procedure billed under {CODE}. Patient, {AGE_NEG} years old, "
    "tolerated the visit well with no acute distress.",

    "Cranial nerves II-XII grossly intact. Coordination normal. "
    "No focal deficits appreciated on this exam.",  # pure negative

    "{VITALS} on arrival, improving to normal range by discharge, "
    "{RELTIME} later.",  # pure negative with distractor numbers

    # second, disjoint-phrased contexts for the four previously
    # single-template classes - eval phrasing intentionally differs from
    # the train phrasing above
    "Front desk verified the patient's SSN ({SSN}) against the chart "
    "before releasing records.",

    "Beneficiary number {HEALTHPLAN} was cross-referenced with the "
    "employer group plan on file.",

    "License plate {VEHICLE} matched the vehicle described in the "
    "incident report.",

    "The recalled device, serial {DEVICE}, was flagged during the "
    "manufacturer's safety audit.",

    "Dispensed by {NAME_TERSE}, verified against the original order.",
    "Order co-signed by {NAME_TERSE} prior to release.",
]


def _render(template: str) -> dict:
    """Fill a template's {PLACEHOLDER} tokens and track char-level entity
    spans as we build the string, including special handling for
    LOCATION_FACILITY (a person's surname embedded inside a LOCATION span,
    but tagged ONLY as LOCATION, never as NAME) and the EPONYM/SURNAME_NAME
    pairing (same surname string, two different correct outcomes)."""
    text = ""
    entities = []
    last_end = 0
    # shared surname per render call so EPONYM <-> SURNAME_NAME pairs can
    # reuse or intentionally diverge; also feeds LOCATION_FACILITY
    shared_surname = _rand_surname()
    eponym_surname = random.choice(list(EPONYM_DISEASES.keys()))

    for match in re.finditer(r"\{(\w+)\}", template):
        label = match.group(1)
        text += template[last_end:match.start()]
        start = len(text)

        if label == "EPONYM":
            value = EPONYM_DISEASES[eponym_surname]
            text += value
            # deliberately NOT added to entities: this is a disease name,
            # not PHI, even though it shares a surname with a physician.
        elif label == "SURNAME_NAME":
            # 50/50: reuse the eponym's surname (the truly hard case) or
            # a random unrelated surname.
            surname = eponym_surname if random.random() < 0.5 else _rand_surname()
            value = f"Dr. {surname}"
            text += value
            entities.append((start, len(text), "NAME"))
        elif label == "LOCATION_FACILITY":
            value = _rand_facility(shared_surname)
            text += value
            entities.append((start, len(text), "LOCATION"))
        elif label == "NAME_REV":
            value = _rand_name_reversed()
            text += value
            entities.append((start, len(text), "NAME"))  # tag as NAME, not NAME_REV
        elif label == "NAME_TERSE":
            value = _rand_name_terse()
            text += value
            entities.append((start, len(text), "NAME"))
        elif label in NEGATIVE_GENERATORS:
            value = NEGATIVE_GENERATORS[label]()
            text += value
            # intentionally not tagged
        else:
            value = FIELD_GENERATORS[label]()
            text += value
            entities.append((start, len(text), label))

        last_end = match.end()
    text += template[last_end:]

    # random casing perturbation so the uncased base model sees signal
    # that doesn't depend on capitalization as a shortcut
    r = random.random()
    if r < 0.15:
        text = text.upper()
    elif r < 0.25:
        text = text.lower()

    return {"text": text, "entities": entities}


def generate_dataset(n_examples=2000, seed=13, split="train"):
    """split='train' draws only from TRAIN_TEMPLATES; split='eval' draws
    only from EVAL_TEMPLATES. These pools are disjoint by construction -
    do not merge them, that was the exact bug in v1."""
    assert split in ("train", "eval")
    pool = TRAIN_TEMPLATES if split == "train" else EVAL_TEMPLATES
    random.seed(seed)
    if HAVE_FAKER:
        Faker.seed(seed)
    return [_render(random.choice(pool)) for _ in range(n_examples)]


if __name__ == "__main__":
    print("=== TRAIN sample ===")
    for ex in generate_dataset(4, split="train"):
        print(ex["text"])
        print(ex["entities"])
        print("-" * 60)
    print("=== EVAL sample (disjoint templates) ===")
    for ex in generate_dataset(4, split="eval", seed=99):
        print(ex["text"])
        print(ex["entities"])
        print("-" * 60)