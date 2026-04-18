from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

SEQ_COL_CANDIDATES = [
    "sequence_tokens",
    "codes",
    "sequence",
    "tokens",
    "event_codes",
    "concepts",
]

BAD_TOKEN_VALUES = {"", "nan", "none", "null", "na", "n/a"}

DIAG_PREFIXES = ("DX_", "ICD", "DIAG_", "SNOMED_")
PROC_PREFIXES = ("PROC_", "CPT_")
MED_PREFIXES = ("MED_", "NDC_", "RXNORM_", "DRUG_", "RAW_DRUG")

# Evidence-based from the actual token namespace in your dataset:
# - male examples with female-specific codes: DX_9_650, DX_10_Z3400
# - female examples with male-specific codes: DX_10_N401, DX_9_185
FEMALE_ONLY_PREFIXES = (
    "DX_9_63",
    "DX_9_64",
    "DX_9_65",
    "DX_9_66",
    "DX_9_67",
    "DX_9_V22",
    "DX_9_V23",
    "DX_9_V24",
    "DX_9_V27",
    "DX_9_V28",
    "DX_10_O",
    "DX_10_Z33",
    "DX_10_Z34",
)

MALE_ONLY_PREFIXES = (
    "DX_9_185",
    "DX_9_600",
    "DX_9_601",
    "DX_9_602",
    "DX_10_C61",
    "DX_10_N40",
)

MEDICATION_SUPPORT_RULES = [
    {
        "name": "insulin_without_diabetes_context",
        "severity": 1.0,
        "med_terms": (
            "INSULIN",
            "GLUCAGON",
            "GLUCOSE_GEL",
            "DEXTROSE_50",
        ),
        "diag_prefixes": (
            "DX_9_250",
            "DX_10_E10",
            "DX_10_E11",
            "DX_10_E13",
            "DX_10_E16",
        ),
        "message": "Diabetes-oriented medication appears without diabetes or glycemic diagnosis context.",
    },
    {
        "name": "bronchodilator_without_respiratory_context",
        "severity": 0.75,
        "med_terms": (
            "ALBUTEROL",
            "MONTELUKAST",
        ),
        "diag_prefixes": (
            "DX_9_49",
            "DX_10_J",
        ),
        "message": "Respiratory medication appears without an obvious respiratory diagnosis context.",
    },
    {
        "name": "antimicrobial_without_infectious_context",
        "severity": 0.75,
        "med_terms": (
            "CIPRO",
            "CEFTRIAXONE",
            "CEFAZOLIN",
            "PENICILLIN",
            "METRONIDAZOLE",
            "VALACYCLOVIR",
            "RIFAXIMIN",
        ),
        "diag_prefixes": (
            "DX_9_0",
            "DX_9_590",
            "DX_9_599",
            "DX_9_682",
            "DX_10_A",
            "DX_10_B",
            "DX_10_J1",
            "DX_10_K65",
            "DX_10_N1",
            "DX_10_R65",
            "DX_10_T81",
            "DX_10_Z20",
        ),
        "message": "Antimicrobial medication appears without an obvious infectious diagnosis context.",
    },
]


@dataclass
class OntologyViolation:
    name: str
    severity: float
    message: str
    implicated_tokens: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            token = str(item).strip()
            if token.lower() not in BAD_TOKEN_VALUES:
                out.append(token)
        return out

    if value is None:
        return []

    text = str(value).strip()
    if text.lower() in BAD_TOKEN_VALUES:
        return []

    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
        if not text:
            return []
        return [part.strip(" '\"") for part in text.split(",") if part.strip(" '\"")]

    return [part for part in text.split() if part.strip()]


def infer_sequence_column(df: Any) -> str:
    for col in SEQ_COL_CANDIDATES:
        if col in df.columns:
            return col

    for col in df.columns:
        sample = df[col].dropna().head(20)
        if len(sample) and any(isinstance(x, (list, tuple)) for x in sample):
            return col

    raise ValueError(f"Could not infer sequence column. Available columns: {list(df.columns)}")


def extract_tokens_from_row(row: Mapping[str, Any] | Any) -> list[str]:
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    if not isinstance(row, Mapping):
        return normalize_tokens(row)

    for col in SEQ_COL_CANDIDATES:
        if col in row:
            return normalize_tokens(row[col])

    return []


def normalize_gender(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"m", "male", "man"}:
        return "male"
    if text in {"f", "female", "woman"}:
        return "female"
    return None


def has_any_prefix(token: str, prefixes: tuple[str, ...]) -> bool:
    up = str(token).upper()
    return any(up.startswith(prefix) for prefix in prefixes)


def token_contains_any(token: str, terms: tuple[str, ...]) -> bool:
    up = str(token).upper()
    return any(term in up for term in terms)


def token_role(token: str) -> str:
    up = str(token).upper()
    if any(up.startswith(prefix) for prefix in DIAG_PREFIXES):
        return "diagnosis"
    if any(up.startswith(prefix) for prefix in PROC_PREFIXES):
        return "procedure"
    if any(up.startswith(prefix) for prefix in MED_PREFIXES):
        return "medication"
    return "other"


def has_supporting_diagnosis(diag_tokens: list[str], diag_prefixes: tuple[str, ...]) -> bool:
    return any(has_any_prefix(token, diag_prefixes) for token in diag_tokens)


def compute_s_ont(record_or_tokens: Mapping[str, Any] | list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if hasattr(record_or_tokens, "to_dict"):
        row = dict(record_or_tokens.to_dict())
    elif isinstance(record_or_tokens, Mapping):
        row = dict(record_or_tokens)

    tokens = extract_tokens_from_row(record_or_tokens) if row else normalize_tokens(record_or_tokens)

    if not tokens:
        return {"sont": 0.0, "violations": [], "token_weights": {}}

    diag_tokens = [tok for tok in tokens if token_role(tok) == "diagnosis"]
    proc_tokens = [tok for tok in tokens if token_role(tok) == "procedure"]
    med_tokens = [tok for tok in tokens if token_role(tok) == "medication"]

    female_only_tokens = [tok for tok in tokens if has_any_prefix(tok, FEMALE_ONLY_PREFIXES)]
    male_only_tokens = [tok for tok in tokens if has_any_prefix(tok, MALE_ONLY_PREFIXES)]

    gender = normalize_gender(row.get("gender") or row.get("sex"))

    violations: list[OntologyViolation] = []

    if med_tokens and not diag_tokens:
        violations.append(
            OntologyViolation(
                name="missing_diagnosis_for_medication",
                severity=1.5,
                message="Medication codes are present but no diagnosis-like code exists in the same record.",
                implicated_tokens=med_tokens[:8],
            )
        )

    if proc_tokens and not diag_tokens:
        violations.append(
            OntologyViolation(
                name="missing_diagnosis_for_procedure",
                severity=1.0,
                message="Procedure codes are present but no diagnosis-like code exists in the same record.",
                implicated_tokens=proc_tokens[:8],
            )
        )

    if gender == "male" and female_only_tokens:
        violations.append(
            OntologyViolation(
                name="male_female_specific_conflict",
                severity=2.5,
                message="Male demographic metadata conflicts with female-specific diagnosis families.",
                implicated_tokens=female_only_tokens[:8],
            )
        )

    if gender == "female" and male_only_tokens:
        violations.append(
            OntologyViolation(
                name="female_male_specific_conflict",
                severity=2.5,
                message="Female demographic metadata conflicts with male-specific diagnosis families.",
                implicated_tokens=male_only_tokens[:8],
            )
        )

    if female_only_tokens and male_only_tokens:
        violations.append(
            OntologyViolation(
                name="forbidden_cooccurrence_demographic_codes",
                severity=2.0,
                message="Male-specific and female-specific diagnosis families co-occur in the same record.",
                implicated_tokens=(female_only_tokens[:4] + male_only_tokens[:4]),
            )
        )

    # Heuristic medication-indication rules.
    # These only fire when diagnosis codes exist, but the relevant support context is missing.
    if diag_tokens:
        for rule in MEDICATION_SUPPORT_RULES:
            matched_meds = [tok for tok in med_tokens if token_contains_any(tok, rule["med_terms"])]
            if matched_meds and not has_supporting_diagnosis(diag_tokens, rule["diag_prefixes"]):
                violations.append(
                    OntologyViolation(
                        name=rule["name"],
                        severity=float(rule["severity"]),
                        message=str(rule["message"]),
                        implicated_tokens=matched_meds[:8],
                    )
                )

    token_weights: dict[str, float] = {}
    for violation in violations:
        unique_tokens = list(dict.fromkeys(violation.implicated_tokens))
        share = violation.severity / max(len(unique_tokens), 1)
        for token in unique_tokens:
            token_weights[token] = token_weights.get(token, 0.0) + share

    return {
        "sont": round(sum(v.severity for v in violations), 6),
        "violations": [v.to_dict() for v in violations],
        "token_weights": {
            k: round(v, 6)
            for k, v in sorted(token_weights.items(), key=lambda x: x[1], reverse=True)
        },
    }
