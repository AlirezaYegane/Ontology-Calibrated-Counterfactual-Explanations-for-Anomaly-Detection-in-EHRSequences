from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from typing import Any


_NULL_STRINGS = {"", "nan", "none", "null", "na", "n/a"}


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        return True
    return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    if _is_nullish(value):
        return default
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if _is_nullish(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _get(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    lowered = {str(k).lower(): k for k in row.keys()}
    for name in names:
        if name in row and not _is_nullish(row[name]):
            return row[name]
        key = lowered.get(name.lower())
        if key is not None and not _is_nullish(row[key]):
            return row[key]
    return default


def _parse_items(value: Any) -> list[str]:
    if _is_nullish(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if not _is_nullish(x)]

    text = str(value).strip()
    if text.lower() in _NULL_STRINGS:
        return []

    if text.startswith("[") and text.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                if isinstance(parsed, list):
                    return _parse_items(parsed)
            except Exception:
                pass

    for sep in ("||", ";;"):
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()]

    if "\n" in text:
        return [x.strip("- ").strip() for x in text.splitlines() if x.strip("- ").strip()]

    return [text]


def _clean_token(token: str) -> str:
    token = str(token).strip()
    token = re.sub(r"\s+", " ", token)
    return token


def _humanize_code_or_action(text: str) -> str:
    text = _clean_token(text)
    text = text.replace("_", " ")
    text = re.sub(r"\bSNOMED\b", "SNOMED", text, flags=re.IGNORECASE)
    text = re.sub(r"\bRXNORM\b", "RxNorm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bICD9\b", "ICD-9", text, flags=re.IGNORECASE)
    text = re.sub(r"\bICD10\b", "ICD-10", text, flags=re.IGNORECASE)
    return text


def _format_list(items: list[str], max_items: int = 4) -> str:
    cleaned = [_humanize_code_or_action(x) for x in items if _clean_token(x)]
    if not cleaned:
        return "not explicitly recorded"
    shown = cleaned[:max_items]
    suffix = "" if len(cleaned) <= max_items else f", plus {len(cleaned) - max_items} more"
    return "; ".join(shown) + suffix


def _format_score(value: float) -> str:
    return f"{value:.4f}"


def _score_reduction(before: float, after: float) -> tuple[float, float]:
    delta = before - after
    pct = 0.0 if before <= 1e-12 else 100.0 * delta / before
    return delta, pct


def _anomaly_label(anomaly_type: str) -> str:
    normalized = anomaly_type.strip().lower().replace("-", "_").replace(" ", "_")
    labels = {
        "demographic_conflict": "a demographic-consistency anomaly",
        "medication_mismatch": "a medication-indication mismatch",
        "missing_diagnosis": "a possible missing-diagnosis / missing-indication anomaly",
        "forbidden_cooccurrence": "a forbidden co-occurrence anomaly",
        "temporal_inconsistency": "a temporal-consistency anomaly",
        "statistical_only": "a mainly statistical anomaly",
        "mixed": "a mixed statistical and ontology-based anomaly",
    }
    return labels.get(normalized, "an anomalous EHR sequence")


def _fallback_violation(anomaly_type: str) -> list[str]:
    normalized = anomaly_type.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "missing_diagnosis":
        return ["the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent"]
    if normalized == "demographic_conflict":
        return ["the sequence contains a code pattern that is inconsistent with the available demographic context"]
    if normalized == "medication_mismatch":
        return ["a medication-related event appears without a sufficiently compatible diagnosis or indication context"]
    if normalized == "forbidden_cooccurrence":
        return ["the sequence contains a pair of events that should not normally co-occur under the ontology rules"]
    if normalized == "temporal_inconsistency":
        return ["the temporal order of events appears inconsistent with the expected clinical sequence"]
    return ["the record was selected by the counterfactual evaluation pipeline, but no specific violation message was stored"]


def _fallback_action(anomaly_type: str) -> list[str]:
    normalized = anomaly_type.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "missing_diagnosis":
        return ["add a compatible missing diagnosis / indication code"]
    if normalized == "demographic_conflict":
        return ["remove or replace the demographic-incompatible code"]
    if normalized == "medication_mismatch":
        return ["add a compatible indication code or revise the mismatched medication event"]
    if normalized == "forbidden_cooccurrence":
        return ["remove or replace one of the conflicting co-occurring codes"]
    if normalized == "temporal_inconsistency":
        return ["revise the event order or remove the temporally inconsistent event"]
    return ["apply the minimal counterfactual edit selected by the search procedure"]


def _primary_driver(
    *,
    s_det: float,
    s_gen: float,
    s_ont: float,
    scal_before: float,
    anomaly_type: str,
    sgen_policy: str,
) -> str:
    normalized = anomaly_type.strip().lower().replace("-", "_").replace(" ", "_")

    ontology_types = {
        "missing_diagnosis",
        "demographic_conflict",
        "medication_mismatch",
        "forbidden_cooccurrence",
        "temporal_inconsistency",
    }

    if s_ont > 0 and s_det > 0:
        return "mixed detector-and-ontology signal"
    if s_ont > 0:
        return "ontology violation signal"
    if s_det > 0:
        return "detector/statistical signal"
    if normalized in ontology_types and scal_before > 0:
        return "ontology-guided counterfactual signal"
    if s_gen > 0 and sgen_policy != "diagnostic_only":
        return "generative surprise signal"
    if scal_before > 0:
        return "calibrated counterfactual score signal"
    return "low-confidence anomaly signal"


def _confidence_phrase(delta: float, edit_count: int) -> str:
    if delta > 0 and edit_count <= 2:
        return "This is a sparse counterfactual explanation because the score improves with only a small number of edits."
    if delta > 0:
        return "The counterfactual reduces the calibrated anomaly score, but it requires more than a minimal one- or two-edit repair."
    return "The counterfactual did not clearly reduce the calibrated score, so this case should be treated as unresolved or low-confidence."


def build_explanation(row: dict[str, Any], *, sgen_policy: str = "diagnostic_only") -> dict[str, Any]:
    record_id = str(
        _get(
            row,
            "record_id",
            "case_id",
            "id",
            "row_id",
            "index",
            "idx",
            "hadm_id",
            "subject_id",
            default="unknown",
        )
    )

    anomaly_type = str(
        _get(
            row,
            "anomaly_type",
            "type",
            "label_name",
            "anomaly_family",
            "violation_type",
            default="unknown",
        )
    )
    anomaly_label = _anomaly_label(anomaly_type)

    s_det = _safe_float(
        _get(
            row,
            "s_det",
            "S_det",
            "sdet",
            "detector_score",
            "prob_anomaly",
            "det_score",
            "supervised_score",
            default=0.0,
        )
    )
    s_gen = _safe_float(
        _get(row, "s_gen", "S_gen", "sgen", "generative_score", "gen_score", default=0.0)
    )
    s_ont = _safe_float(
        _get(
            row,
            "s_ont",
            "S_ont",
            "sont",
            "ontology_score",
            "ontology_violation_score",
            "violation_score",
            "violation_count",
            "raw_original_violation_score",
            "violations_before",
            default=0.0,
        )
    )

    scal_before = _safe_float(
        _get(
            row,
            "s_cal_before",
            "S_cal_before",
            "scal_before",
            "original_scal",
            "original_s_cal",
            "score_before",
            "original_score",
            "score_original",
            "s_cal",
            "S_cal",
            "scal",
            default=0.0,
        )
    )
    scal_after = _safe_float(
        _get(
            row,
            "s_cal_after",
            "S_cal_after",
            "scal_after",
            "counterfactual_scal",
            "counterfactual_score",
            "cf_score",
            "cf_scal",
            "score_after",
            "best_score",
            "S_cal_star",
            default=scal_before,
        )
    )

    violations = _parse_items(
        _get(
            row,
            "violations",
            "violation_messages",
            "issues",
            "ontology_violations",
            "violation",
            "issue",
            "reason",
            "repair_reason",
            default=[],
        )
    )
    actions = _parse_items(
        _get(
            row,
            "actions",
            "action",
            "edit_action",
            "edit_sequence",
            "applied_edits",
            "best_action",
            "repair_action",
            "counterfactual_action",
            "operation",
            "edit_operation",
            "action_raw",
            "raw_edits_text",
            default=[],
        )
    )

    if not violations:
        violations = _fallback_violation(anomaly_type)
    if not actions:
        actions = _fallback_action(anomaly_type)

    edit_count = _safe_int(_get(row, "edit_count", "n_edits", "num_edits", "num_operations", default=len(actions)))
    if edit_count == 0 and actions:
        edit_count = len(actions)

    delta, pct = _score_reduction(scal_before, scal_after)
    driver = _primary_driver(
        s_det=s_det,
        s_gen=s_gen,
        s_ont=s_ont,
        scal_before=scal_before,
        anomaly_type=anomaly_type,
        sgen_policy=sgen_policy,
    )

    violation_text = _format_list(violations)
    action_text = _format_list(actions)

    sgen_sentence = (
        "The generative surprise score is reported as a diagnostic auxiliary signal only, "
        "because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation."
        if sgen_policy == "diagnostic_only"
        else "The generative surprise score is included as one component of the explanation."
    )

    short = (
        f"Record {record_id} was flagged as {anomaly_label}. "
        f"The main evidence comes from the {driver}. "
        f"The proposed counterfactual repair is: {action_text}. "
        f"This changes the calibrated score from {_format_score(scal_before)} to {_format_score(scal_after)} "
        f"(ΔScal={_format_score(delta)}, {pct:.1f}% reduction)."
    )

    clinical = (
        f"Record {record_id} appears unusual because {violation_text}. "
        f"The proposed counterfactual is to {action_text}. "
        f"After this edit, the calibrated anomaly score changes from {_format_score(scal_before)} "
        f"to {_format_score(scal_after)}. "
        f"{_confidence_phrase(delta, edit_count)} "
        f"This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation."
    )

    research = (
        f"For case {record_id}, the explanation generator classifies the example as {anomaly_label}. "
        f"The decomposed scores available to the text generator are Sdet={_format_score(s_det)}, "
        f"Sgen={_format_score(s_gen)}, Sont={_format_score(s_ont)}, and "
        f"Scal_before={_format_score(scal_before)}. "
        f"The selected counterfactual applies {edit_count} edit(s): {action_text}. "
        f"The resulting score is Scal_after={_format_score(scal_after)}, giving "
        f"ΔScal={_format_score(delta)} ({pct:.1f}% relative reduction). "
        f"Ontology/counterfactual evidence: {violation_text}. {sgen_sentence}"
    )

    return {
        "record_id": record_id,
        "anomaly_type": anomaly_type,
        "primary_driver": driver,
        "edit_count": edit_count,
        "s_det": s_det,
        "s_gen": s_gen,
        "s_ont": s_ont,
        "scal_before": scal_before,
        "scal_after": scal_after,
        "delta_scal": delta,
        "relative_reduction_pct": pct,
        "violations_compact": violation_text,
        "actions_compact": action_text,
        "explanation_short": short,
        "explanation_clinical": clinical,
        "explanation_research": research,
    }


def build_explanation_batch(rows: list[dict[str, Any]], *, sgen_policy: str = "diagnostic_only") -> list[dict[str, Any]]:
    return [build_explanation(row, sgen_policy=sgen_policy) for row in rows]


def summarize_explanations(explanations: list[dict[str, Any]]) -> dict[str, Any]:
    if not explanations:
        return {
            "n_cases": 0,
            "mean_delta_scal": 0.0,
            "median_delta_scal": 0.0,
            "pct_positive_reduction": 0.0,
            "pct_one_or_two_edits": 0.0,
            "driver_counts": {},
            "anomaly_type_counts": {},
        }

    deltas = sorted(float(x["delta_scal"]) for x in explanations)
    mid = len(deltas) // 2
    median = deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2.0

    driver_counts = Counter(str(x["primary_driver"]) for x in explanations)
    anomaly_counts = Counter(str(x["anomaly_type"]) for x in explanations)

    return {
        "n_cases": len(explanations),
        "mean_delta_scal": sum(deltas) / len(deltas),
        "median_delta_scal": median,
        "pct_positive_reduction": sum(1 for x in explanations if float(x["delta_scal"]) > 0) / len(explanations),
        "pct_one_or_two_edits": sum(1 for x in explanations if int(x["edit_count"]) in {1, 2}) / len(explanations),
        "driver_counts": dict(driver_counts),
        "anomaly_type_counts": dict(anomaly_counts),
    }
