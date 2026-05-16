from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


BAD_VALUES = {"", "nan", "none", "null", "na", "n/a"}

SEQUENCE_COLUMNS = (
    "sequence_tokens",
    "codes",
    "tokens",
    "sequence",
    "event_codes",
    "concepts",
    "record",
)

ANOMALY_TYPE_COLUMNS = (
    "anomaly_type",
    "type",
    "violation_type",
    "issue_type",
)

SEX_COLUMNS = (
    "sex",
    "gender",
    "patient_sex",
)

EXPECTED_CODE_COLUMNS = (
    "expected_code",
    "expected_codes",
    "missing_code",
    "missing_codes",
    "removed_code",
    "removed_codes",
    "required_code",
    "required_codes",
    "target_code",
    "target_codes",
    "indication_code",
    "indication_codes",
)

BAD_CODE_COLUMNS = (
    "bad_code",
    "bad_codes",
    "injected_code",
    "injected_codes",
    "added_code",
    "added_codes",
    "conflict_code",
    "conflict_codes",
    "flagged_code",
    "flagged_codes",
    "problem_code",
    "problem_codes",
)

REPLACEMENT_COLUMNS = (
    "replacement_code",
    "replacement_codes",
    "suggested_replacement",
    "suggested_replacements",
)

PREGNANCY_PATTERNS = (
    "PREG",
    "PREGNAN",
    "OBSTET",
    "GESTATION",
    "ANTENATAL",
    "DELIVERY",
    "CHILDBIRTH",
    "PUERPER",
    "MATERNAL",
    "ICD10_O",
    "ICD10:O",
    "ICD9_V22",
    "ICD9:V22",
    "ICD9_V23",
    "ICD9:V23",
    "ICD9_63",
    "ICD9:63",
)

DIAG_PREFIXES = (
    "ICD",
    "ICD9",
    "ICD10",
    "DIAG",
    "DX",
    "SNOMED",
)

MED_PREFIXES = (
    "MED",
    "DRUG",
    "RX",
    "RXNORM",
    "NDC",
    "RAW_DRUG",
)

PROC_PREFIXES = (
    "PROC",
    "CPT",
    "PROCEDURE",
)


@dataclass(frozen=True)
class EditOperation:
    kind: str
    code: str | None = None
    new_code: str | None = None
    reason: str = ""

    def label(self) -> str:
        if self.kind == "remove":
            return f"remove {self.code}"
        if self.kind == "add":
            return f"add {self.new_code}"
        if self.kind == "replace":
            return f"replace {self.code} -> {self.new_code}"
        return self.kind


@dataclass(frozen=True)
class Violation:
    violation_type: str
    message: str
    codes: tuple[str, ...] = ()
    expected_codes: tuple[str, ...] = ()
    severity: float = 1.0


@dataclass
class CounterfactualResult:
    original_codes: list[str]
    counterfactual_codes: list[str]
    edits: list[EditOperation]
    original_violations: list[Violation]
    counterfactual_violations: list[Violation]
    original_violation_score: float
    counterfactual_violation_score: float
    cost: float
    status: str

    @property
    def edit_count(self) -> int:
        return len(self.edits)

    @property
    def delta_violation_score(self) -> float:
        return self.original_violation_score - self.counterfactual_violation_score

    @property
    def resolved_violation_count(self) -> int:
        return max(
            0, len(self.original_violations) - len(self.counterfactual_violations)
        )

    def edits_as_text(self) -> str:
        if not self.edits:
            return ""
        return "; ".join(edit.label() for edit in self.edits)

    def explanation(self) -> str:
        if self.status == "improved":
            return (
                f"Counterfactual improved the ontology score by "
                f"{self.delta_violation_score:.4f} using {self.edit_count} edit(s): "
                f"{self.edits_as_text()}."
            )
        if self.status == "no_candidate":
            return "No safe ontology-constrained edit candidate was generated for this record."
        return "Candidate edits were generated, but none reduced the ontology violation score."


def normalize_token(value: Any) -> str:
    return str(value).strip()


def is_bad_value(value: Any) -> bool:
    return normalize_token(value).lower() in BAD_VALUES


def parse_list_like(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, float) and str(value).lower() == "nan":
        return []

    if isinstance(value, (list, tuple, set)):
        return [normalize_token(x) for x in value if not is_bad_value(x)]

    text = normalize_token(value)
    if text.lower() in BAD_VALUES:
        return []

    if text.startswith("[") and text.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(text)
                if isinstance(obj, (list, tuple, set)):
                    return parse_list_like(obj)
            except Exception:
                pass

    if text.startswith("{") and text.endswith("}"):
        return []

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]

    return [text]


def first_existing_key(row: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    lower_to_real = {str(k).lower(): str(k) for k in row.keys()}
    for key in keys:
        if key.lower() in lower_to_real:
            return lower_to_real[key.lower()]
    return None


def get_many_from_columns(row: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        real_key = first_existing_key(row, (key,))
        if real_key is not None:
            values.extend(parse_list_like(row.get(real_key)))
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def infer_sequence_column(
    row_or_columns: Mapping[str, Any] | Sequence[str],
) -> str | None:
    columns = (
        row_or_columns.keys() if isinstance(row_or_columns, Mapping) else row_or_columns
    )
    lower_to_real = {str(c).lower(): str(c) for c in columns}
    for col in SEQUENCE_COLUMNS:
        if col.lower() in lower_to_real:
            return lower_to_real[col.lower()]
    return None


def tokens_from_row(row: Mapping[str, Any]) -> list[str]:
    seq_col = infer_sequence_column(row)
    if seq_col is None:
        return []
    return parse_list_like(row.get(seq_col))


def anomaly_type_from_row(row: Mapping[str, Any]) -> str:
    col = first_existing_key(row, ANOMALY_TYPE_COLUMNS)
    if col is None:
        return ""
    values = parse_list_like(row.get(col))
    return values[0].strip().lower() if values else ""


def sex_from_row(row: Mapping[str, Any]) -> str:
    col = first_existing_key(row, SEX_COLUMNS)
    if col is None:
        return ""
    values = parse_list_like(row.get(col))
    return values[0].strip().lower() if values else ""


def token_upper(token: str) -> str:
    return token.strip().upper()


def is_pregnancy_token(token: str) -> bool:
    up = token_upper(token)
    return any(pattern in up for pattern in PREGNANCY_PATTERNS)


def is_medication_token(token: str) -> bool:
    up = token_upper(token)
    return up.startswith(MED_PREFIXES) or any(
        up.startswith(f"{p}_") or up.startswith(f"{p}:") for p in MED_PREFIXES
    )


def is_diagnosis_token(token: str) -> bool:
    up = token_upper(token)
    return up.startswith(DIAG_PREFIXES) or any(
        up.startswith(f"{p}_") or up.startswith(f"{p}:") for p in DIAG_PREFIXES
    )


def is_procedure_token(token: str) -> bool:
    up = token_upper(token)
    return up.startswith(PROC_PREFIXES) or any(
        up.startswith(f"{p}_") or up.startswith(f"{p}:") for p in PROC_PREFIXES
    )


def expected_codes_from_row(row: Mapping[str, Any]) -> list[str]:
    return get_many_from_columns(row, EXPECTED_CODE_COLUMNS)


def bad_codes_from_row(row: Mapping[str, Any], codes: Sequence[str]) -> list[str]:
    explicit = get_many_from_columns(row, BAD_CODE_COLUMNS)
    sex = sex_from_row(row)
    anomaly_type = anomaly_type_from_row(row)

    inferred: list[str] = []
    if sex.startswith("m") or "male" in sex or anomaly_type == "demographic_conflict":
        inferred.extend([code for code in codes if is_pregnancy_token(code)])

    out: list[str] = []
    seen: set[str] = set()
    for code in [*explicit, *inferred]:
        if code in codes and code not in seen:
            seen.add(code)
            out.append(code)

    return out


def replacement_codes_from_row(row: Mapping[str, Any]) -> list[str]:
    return get_many_from_columns(row, REPLACEMENT_COLUMNS)


def detect_violations(codes: Sequence[str], row: Mapping[str, Any]) -> list[Violation]:
    anomaly_type = anomaly_type_from_row(row)
    sex = sex_from_row(row)
    expected_codes = expected_codes_from_row(row)
    bad_codes = bad_codes_from_row(row, codes)

    code_set = set(codes)
    violations: list[Violation] = []

    preg_codes = [code for code in codes if is_pregnancy_token(code)]
    if preg_codes and (
        sex.startswith("m") or "male" in sex or anomaly_type == "demographic_conflict"
    ):
        violations.append(
            Violation(
                violation_type="demographic_conflict",
                message="Pregnancy/obstetric code appears in a male or demographic-conflict record.",
                codes=tuple(preg_codes),
                severity=2.0,
            )
        )

    # Repair-ready Day 36 data provides explicit bad_code evidence reconstructed
    # from codes_original vs. codes_corrupted. This is more reliable than trying
    # to infer every demographic rule from token text alone, especially for
    # prostate-in-female and pregnancy-in-male synthetic corruptions.
    if anomaly_type == "demographic_conflict" and bad_codes:
        already_flagged = {code for violation in violations for code in violation.codes}
        explicit_bad_codes = [code for code in bad_codes if code not in already_flagged]
        if explicit_bad_codes:
            violations.append(
                Violation(
                    violation_type="demographic_conflict",
                    message="Row-marked demographic-incompatible code is present.",
                    codes=tuple(explicit_bad_codes),
                    severity=2.0,
                )
            )

    missing_expected = [code for code in expected_codes if code not in code_set]
    if missing_expected:
        vtype = (
            "missing_diagnosis"
            if anomaly_type == "missing_diagnosis"
            else "missing_expected_code"
        )
        violations.append(
            Violation(
                violation_type=vtype,
                message="Expected ontology-related code is absent from the record.",
                expected_codes=tuple(missing_expected),
                severity=1.5 if vtype == "missing_diagnosis" else 1.0,
            )
        )

    if anomaly_type in {"medication_mismatch", "missing_indication"}:
        has_med = any(is_medication_token(code) for code in codes)
        has_diag = any(is_diagnosis_token(code) for code in codes)
        if has_med and not has_diag and not expected_codes:
            violations.append(
                Violation(
                    violation_type="medication_mismatch",
                    message="Medication-like code appears without a detectable diagnosis/indication token.",
                    codes=tuple(code for code in codes if is_medication_token(code)),
                    severity=1.0,
                )
            )

    if (
        anomaly_type in {"forbidden_pair", "pairwise_conflict", "contradiction"}
        and bad_codes
    ):
        violations.append(
            Violation(
                violation_type="forbidden_pair",
                message="A row-marked conflicting code is present.",
                codes=tuple(bad_codes),
                severity=1.25,
            )
        )

    return violations


def violation_score(violations: Sequence[Violation]) -> float:
    return float(sum(v.severity for v in violations))


def remove_code(codes: Sequence[str], code: str) -> list[str]:
    removed = False
    out: list[str] = []
    for item in codes:
        if item == code and not removed:
            removed = True
            continue
        out.append(item)
    return out


def add_code(codes: Sequence[str], code: str) -> list[str]:
    if code in codes:
        return list(codes)
    return [*codes, code]


def replace_code(codes: Sequence[str], old: str, new: str) -> list[str]:
    replaced = False
    out: list[str] = []
    for item in codes:
        if item == old and not replaced:
            out.append(new)
            replaced = True
        else:
            out.append(item)
    return out


def apply_edit(codes: Sequence[str], edit: EditOperation) -> list[str]:
    if edit.kind == "remove" and edit.code is not None:
        return remove_code(codes, edit.code)
    if edit.kind == "add" and edit.new_code is not None:
        return add_code(codes, edit.new_code)
    if edit.kind == "replace" and edit.code is not None and edit.new_code is not None:
        return replace_code(codes, edit.code, edit.new_code)
    return list(codes)


def apply_edits(codes: Sequence[str], edits: Sequence[EditOperation]) -> list[str]:
    out = list(codes)
    for edit in edits:
        out = apply_edit(out, edit)
    return out


def propose_one_step_edits(
    codes: Sequence[str], row: Mapping[str, Any]
) -> list[EditOperation]:
    edits: list[EditOperation] = []

    for bad_code in bad_codes_from_row(row, codes):
        edits.append(
            EditOperation(
                kind="remove",
                code=bad_code,
                reason="remove row-marked or inferred ontology-conflicting code",
            )
        )

    expected_codes = expected_codes_from_row(row)
    for expected in expected_codes:
        if expected not in codes:
            edits.append(
                EditOperation(
                    kind="add",
                    new_code=expected,
                    reason="add missing ontology-expected code",
                )
            )

    replacements = replacement_codes_from_row(row)
    bad_codes = bad_codes_from_row(row, codes)
    for bad_code in bad_codes:
        for replacement in replacements:
            if replacement != bad_code and replacement not in codes:
                edits.append(
                    EditOperation(
                        kind="replace",
                        code=bad_code,
                        new_code=replacement,
                        reason="replace conflicting code with suggested ontology-neighbour",
                    )
                )

    deduped: list[EditOperation] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for edit in edits:
        key = (edit.kind, edit.code, edit.new_code)
        if key not in seen:
            seen.add(key)
            deduped.append(edit)

    return deduped


def generate_counterfactual(
    codes: Sequence[str],
    row: Mapping[str, Any],
    edit_penalty: float = 0.25,
    max_edits: int = 2,
    score_fn: Callable[[Sequence[str], Mapping[str, Any]], float] | None = None,
) -> CounterfactualResult:
    score_fn = score_fn or (
        lambda candidate_codes, context: violation_score(
            detect_violations(candidate_codes, context)
        )
    )

    original_codes = list(codes)
    original_violations = detect_violations(original_codes, row)
    original_score = score_fn(original_codes, row)

    one_step = propose_one_step_edits(original_codes, row)
    if not one_step:
        return CounterfactualResult(
            original_codes=original_codes,
            counterfactual_codes=original_codes,
            edits=[],
            original_violations=original_violations,
            counterfactual_violations=original_violations,
            original_violation_score=original_score,
            counterfactual_violation_score=original_score,
            cost=original_score,
            status="no_candidate",
        )

    candidate_edit_sets: list[list[EditOperation]] = [[edit] for edit in one_step]

    if max_edits >= 2:
        for i, first in enumerate(one_step):
            after_first = apply_edits(original_codes, [first])
            second_steps = propose_one_step_edits(after_first, row)
            for second in second_steps:
                if second == first:
                    continue
                candidate_edit_sets.append([first, second])

    best_edits: list[EditOperation] = []
    best_codes = original_codes
    best_score = original_score
    best_cost = original_score

    for edits in candidate_edit_sets:
        candidate_codes = apply_edits(original_codes, edits)
        candidate_score = score_fn(candidate_codes, row)
        candidate_cost = candidate_score + edit_penalty * len(edits)

        if candidate_score < best_score or (
            candidate_score == best_score and candidate_cost < best_cost
        ):
            best_score = candidate_score
            best_cost = candidate_cost
            best_codes = candidate_codes
            best_edits = edits

    counterfactual_violations = detect_violations(best_codes, row)
    status = "improved" if best_score < original_score else "not_improved"

    return CounterfactualResult(
        original_codes=original_codes,
        counterfactual_codes=best_codes,
        edits=best_edits,
        original_violations=original_violations,
        counterfactual_violations=counterfactual_violations,
        original_violation_score=original_score,
        counterfactual_violation_score=best_score,
        cost=best_cost,
        status=status,
    )


def result_to_dict(result: CounterfactualResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "edit_count": result.edit_count,
        "edits": [asdict(edit) for edit in result.edits],
        "edits_text": result.edits_as_text(),
        "original_violation_score": result.original_violation_score,
        "counterfactual_violation_score": result.counterfactual_violation_score,
        "delta_violation_score": result.delta_violation_score,
        "resolved_violation_count": result.resolved_violation_count,
        "original_violations": [asdict(v) for v in result.original_violations],
        "counterfactual_violations": [
            asdict(v) for v in result.counterfactual_violations
        ],
        "original_codes": result.original_codes,
        "counterfactual_codes": result.counterfactual_codes,
        "explanation": result.explanation(),
    }


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
