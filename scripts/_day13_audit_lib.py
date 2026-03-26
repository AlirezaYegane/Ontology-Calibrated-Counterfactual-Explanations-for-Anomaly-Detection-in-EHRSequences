from __future__ import annotations

import ast
import csv
import itertools
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TEXT_FILE_SUFFIXES = {".jsonl", ".json", ".csv", ".pkl", ".parquet"}
REPORT_BASENAMES = {
    "mapping_audit.json",
    "mapping_audit.md",
    "edge_case_report.json",
}

VALUE_KEYS = {
    "code",
    "codes",
    "token",
    "tokens",
    "sequence",
    "sequences",
    "events",
    "event_codes",
    "concept",
    "concepts",
    "diagnosis_codes",
    "procedure_codes",
    "medication_codes",
    "diag_codes",
    "proc_codes",
    "med_codes",
}

NON_CODE_STRING_KEYS = {
    "description",
    "descriptions",
    "title",
    "name",
    "text",
    "note",
    "notes",
    "label",
    "labels",
    "long_title",
    "short_title",
    "display",
    "display_name",
}

DEMOGRAPHIC_KEYS = {
    "sex",
    "gender",
    "age",
    "anchor_age",
    "dob",
    "anchor_year",
}

METADATA_KEYS = {
    "subject_id",
    "hadm_id",
    "patient_id",
    "patientunitstayid",
    "stay_ids",
    "admittime",
    "dischtime",
    "split",
    "sequence_length",
    "n_diagnoses",
    "n_procedures",
    "n_medications",
    "n_comorbidities",
    "hospital_los_days",
    "icu_los_days",
    "hospital_death",
    "icu_death",
    "ethnicity",
    "age_years",
    "gender",
    "age_group",
}

SINGLE_TOKEN_KEYS = {
    "token",
    "tokens",
    "sequence",
    "sequences",
    "event_codes",
    "concept",
    "concepts",
}

TOKEN_LIKE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
LOWERCASE_NAMESPACE_RE = re.compile(r"^[a-z]+[:_]")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    return str(value).strip()


def is_empty_token(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"", "nan", "none", "null", "na", "n/a"}


def infer_category_from_key(key: str | None) -> str | None:
    if not key:
        return None
    key_l = key.lower()
    if "diag" in key_l or key_l in {"dx", "dxs"}:
        return "diagnosis"
    if "proc" in key_l or "operation" in key_l or "cpt" in key_l:
        return "procedure"
    if any(x in key_l for x in ("med", "drug", "rx", "prescription", "ndc", "formulary")):
        return "medication"
    if any(x in key_l for x in ("token", "sequence", "event", "concept")):
        return "sequence"
    return None


def infer_category_from_token(token: str) -> str:
    up = token.upper()
    if up.startswith(("ICD", "DX", "DIAG", "SNOMED")):
        return "diagnosis"
    if up.startswith(("PROC", "CPT")):
        return "procedure"
    if up.startswith(("RXNORM", "NDC", "MED", "DRUG", "RAW_DRUG")):
        return "medication"
    return "other"


def namespace_from_token(token: str) -> str:
    up = token.upper()
    if up.startswith("SNOMED"):
        return "SNOMED"
    if up.startswith("RXNORM"):
        return "RXNORM"
    if up.startswith("NDC"):
        return "NDC"
    if up.startswith(("ICD9", "ICD10", "ICD")):
        return "ICD"
    if up.startswith(("PROC", "CPT")):
        return "PROC"
    if up.startswith(("MED", "DRUG", "RAW_DRUG")):
        return "MED"
    if up.startswith(("UNMAPPED", "UNKNOWN", "UNK", "RAW_")):
        return "UNMAPPED"
    return "OTHER"


def is_unmapped_token(token: str) -> bool:
    """Check if a token indicates an unmapped or unknown concept."""
    up = token.upper()
    return (
        up.startswith(("UNMAPPED", "UNKNOWN", "UNK", "RAW_", "RAW-", "NO_MAP"))
        or ":UNMAPPED" in up
        or up.startswith("RAW_DRUG")
    )


def is_multi_mapped_like(token: str) -> bool:
    return "|" in token or ";" in token


def has_lowercase_namespace(token: str) -> bool:
    return bool(LOWERCASE_NAMESPACE_RE.match(token))


def is_malformed_token(token: str) -> bool:
    if is_empty_token(token):
        return False
    if is_multi_mapped_like(token):
        return False
    return not bool(TOKEN_LIKE_RE.fullmatch(token))


def looks_tokenish(text: str) -> bool:
    if is_empty_token(text):
        return True
    if len(text) > 80:
        return False
    if bool(TOKEN_LIKE_RE.fullmatch(text)):
        return True
    if "_" in text or ":" in text or text.isdigit():
        return True
    return False


def maybe_split_string(text: str) -> list[str]:
    text = normalize_text(text)
    if text == "":
        return [""]
    if "," in text:
        return [part.strip() for part in text.split(",")]
    if " " in text:
        parts = text.split()
        if parts and all(looks_tokenish(part) for part in parts):
            return parts
    return [text]


def parse_list_like_string(text: str) -> list[str] | None:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None

    if text == "[]":
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(text)
            if isinstance(obj, list):
                return [normalize_text(item) for item in obj]
        except Exception:
            continue

    return None


def collect_observations(
    value: Any,
    current_key: str = "",
    category: str | None = None,
    allow_scalars: bool = False,
) -> list[dict[str, str]]:
    observations: list[dict[str, str]] = []

    if isinstance(value, dict):
        keys_lower = {str(k).lower() for k in value.keys()}
        redundant_keys: set[str] = set()
        if keys_lower & {
            "diagnosis_tokens",
            "procedure_tokens",
            "medication_tokens",
            "treatment_tokens",
            "comorbidity_tokens",
        }:
            redundant_keys.update({"sequence_tokens", "codes"})

        for key, subvalue in value.items():
            key_s = str(key)
            key_l = key_s.lower()
            if key_l in redundant_keys:
                continue

            inferred_cat = infer_category_from_key(key_s)
            next_category = inferred_cat or category
            next_allow = allow_scalars or (inferred_cat is not None) or (key_l in VALUE_KEYS)
            observations.extend(
                collect_observations(
                    subvalue,
                    current_key=key_s,
                    category=next_category,
                    allow_scalars=next_allow,
                )
            )
        return observations

    if isinstance(value, (list, tuple, set)):
        for item in value:
            observations.extend(
                collect_observations(
                    item,
                    current_key=current_key,
                    category=category,
                    allow_scalars=True,
                )
            )
        return observations

    if isinstance(value, (str, int, float)):
        key_l = current_key.lower()
        if key_l in NON_CODE_STRING_KEYS or key_l in METADATA_KEYS:
            return []
        if not (allow_scalars or key_l in VALUE_KEYS or current_key == ""):
            return []

        text = normalize_text(value)
        parsed_list = parse_list_like_string(text)
        if parsed_list is not None:
            return collect_observations(
                parsed_list,
                current_key=current_key,
                category=category,
                allow_scalars=True,
            )

        preserve_as_single_token = category == "sequence" or key_l in SINGLE_TOKEN_KEYS
        parts = [text] if preserve_as_single_token else maybe_split_string(text)
        resolved_category = category or infer_category_from_key(current_key) or "other"

        for part in parts:
            item_category = resolved_category
            if item_category == "sequence":
                item_category = infer_category_from_token(part)
            observations.append(
                {
                    "token": part,
                    "category": item_category,
                    "source_key": current_key or "root",
                }
            )
        return observations

    return observations


def extract_items(obj: Any) -> Iterator[Any]:
    """Helper to extract iterables from common metadata-wrapped dictionary structures."""
    if isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for container_key in ("records", "rows", "data", "items", "examples"):
            maybe_items = obj.get(container_key)
            if isinstance(maybe_items, list):
                yield from maybe_items
                return
        yield obj
    else:
        yield obj


def iter_records(path: Path, max_records: int | None = None) -> Iterator[Any]:
    """Yield records iteratively from standard data artifacts up to max_records."""
    suffix = path.suffix.lower()

    def _limit(iterable: Iterable[Any]) -> Iterator[Any]:
        if max_records is None:
            yield from iterable
        else:
            yield from itertools.islice(iterable, max_records)

    try:
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="ignore") as fh:

                def _gen() -> Any:
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield json.loads(line)

                yield from _limit(_gen())

        elif suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            yield from _limit(extract_items(obj))

        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
                yield from _limit(csv.DictReader(fh))

        elif suffix in {".pkl", ".parquet"}:
            import pandas as pd

            obj = pd.read_pickle(path) if suffix == ".pkl" else pd.read_parquet(path)

            if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
                yield from _limit(obj.to_dict(orient="records"))
            else:
                yield from _limit(extract_items(obj))
        else:
            raise RuntimeError(f"Unsupported object type from {path.as_posix()}: {suffix}")

    except Exception as exc:
        raise RuntimeError(f"Failed to load {path.as_posix()}: {exc}") from exc


def discover_processed_files(processed_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not processed_dir.exists():
        return files

    include_patterns = (
        "train",
        "test",
        "val",
        "valid",
        "sequence",
        "sequences",
        "dataset",
    )
    exclude_patterns = (
        "stats",
        "summary",
        "manifest",
        "split_ids",
        "mapping_audit",
        "edge_case_report",
    )

    for path in processed_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in REPORT_BASENAMES:
            continue
        if path.suffix.lower() not in TEXT_FILE_SUFFIXES:
            continue

        name_l = path.name.lower()
        if any(pattern in name_l for pattern in exclude_patterns):
            continue
        if not any(pattern in name_l for pattern in include_patterns):
            continue

        files.append(path)

    return sorted(files)


def round_rate(mapped: int, total: int) -> float | None:
    """Calculate the mapping rate precisely, rounded to 4 decimals."""
    if total <= 0:
        return None
    return round(mapped / total, 4)


def build_audit_report(
    processed_dir: Path,
    max_records_per_file: int | None = 2000,
) -> dict[str, Any]:
    """End-to-end scanner generating the main dict for the Day 13 mapping audit."""
    processed_dir = Path(processed_dir)
    files = discover_processed_files(processed_dir)

    critical_issues: list[str] = []
    warnings: list[str] = []
    files_checked: list[dict[str, Any]] = []

    mapping_counts: dict[str, dict[str, int]] = {
        "diagnosis": {"total": 0, "mapped": 0, "unmapped": 0},
        "procedure": {"total": 0, "mapped": 0, "unmapped": 0},
        "medication": {"total": 0, "mapped": 0, "unmapped": 0},
        "other": {"total": 0, "mapped": 0, "unmapped": 0},
    }
    top_unmapped: dict[str, Counter[str]] = {
        "diagnosis": Counter(),
        "procedure": Counter(),
        "medication": Counter(),
        "other": Counter(),
    }
    namespace_summary: Counter[str] = Counter()
    edge_case_counts: Counter[str] = Counter()

    MAX_EXAMPLES = 25

    unknown_namespace_examples: Counter[str] = Counter()
    malformed_examples: Counter[str] = Counter()
    other_examples: Counter[str] = Counter()
    duplicate_examples: Counter[str] = Counter()

    total_records = 0
    total_tokens = 0
    split_like_files: list[str] = []

    if not processed_dir.exists():
        critical_issues.append(
            f"Processed directory does not exist: {processed_dir.as_posix()}"
        )

    for path in files:
        file_record_count = 0

        try:
            for record in iter_records(path, max_records=max_records_per_file):
                file_record_count += 1
                total_records += 1

                if isinstance(record, dict):
                    record_keys = {str(k).lower() for k in record.keys()}
                    if DEMOGRAPHIC_KEYS.isdisjoint(record_keys):
                        edge_case_counts["records_missing_demographics"] += 1

                observations = collect_observations(record)
                if not observations:
                    edge_case_counts["records_without_observations"] += 1

                seen_tokens: Counter[str] = Counter()
                cleaned_nonempty = 0

                for obs in observations:
                    token = normalize_text(obs["token"])

                    if is_empty_token(token):
                        edge_case_counts["empty_tokens"] += 1
                        continue

                    category = obs["category"]
                    if category not in mapping_counts:
                        category = infer_category_from_token(token)
                        if category not in mapping_counts:
                            category = "other"

                    total_tokens += 1
                    cleaned_nonempty += 1

                    duplicate_key = token.upper()
                    seen_tokens[duplicate_key] += 1

                    mapping_counts[category]["total"] += 1
                    if is_unmapped_token(token):
                        mapping_counts[category]["unmapped"] += 1
                        top_unmapped[category][token] += 1
                    else:
                        mapping_counts[category]["mapped"] += 1

                    namespace = namespace_from_token(token)
                    namespace_summary[namespace] += 1

                    if namespace == "OTHER":
                        edge_case_counts["unknown_namespace_tokens"] += 1
                        unknown_namespace_examples[token] += 1
                        other_examples[token] += 1

                    if is_malformed_token(token):
                        edge_case_counts["malformed_tokens"] += 1
                        malformed_examples[token] += 1

                    if has_lowercase_namespace(token):
                        edge_case_counts["lowercase_namespace_tokens"] += 1

                    if is_multi_mapped_like(token):
                        edge_case_counts["multi_mapped_like_tokens"] += 1

                for dup_token, dup_count in seen_tokens.items():
                    if dup_count > 1:
                        duplicate_examples[dup_token] += dup_count

                edge_case_counts["duplicate_tokens_within_record"] += sum(
                    count - 1 for count in seen_tokens.values() if count > 1
                )

                if cleaned_nonempty == 0:
                    edge_case_counts["records_empty_after_cleaning"] += 1

        except Exception as exc:
            critical_issues.append(str(exc))

        files_checked.append(
            {
                "path": path.as_posix(),
                "records_loaded": file_record_count,
                "sampled": max_records_per_file is not None,
            }
        )

        if re.search(r"(train|valid|val|test|split)", path.name, re.IGNORECASE):
            split_like_files.append(path.as_posix())

    if not files:
        critical_issues.append(
            f"No loadable .jsonl/.json/.csv files found under {processed_dir.as_posix()}"
        )

    if total_records == 0:
        critical_issues.append("No records were loaded from processed artifacts.")

    if total_tokens == 0:
        critical_issues.append("No code/token observations were extracted from processed artifacts.")

    if not split_like_files:
        warnings.append("No split-like artifacts detected via filename pattern: train/val/test/split")

    mapping_summary: dict[str, dict[str, Any]] = {}
    for category, counts in mapping_counts.items():
        total = counts["total"]
        mapped = counts["mapped"]
        unmapped = counts["unmapped"]
        mapping_summary[category] = {
            "total": total,
            "mapped": mapped,
            "unmapped": unmapped,
            "mapping_rate": round_rate(mapped, total),
        }
        if total > 0 and mapped == 0:
            warnings.append(f"{category} category has observations but 0 mapped tokens.")

    milestone1_ready = len(critical_issues) == 0 and total_records > 0 and total_tokens > 0

    report: dict[str, Any] = {
        "generated_at": now_utc_iso(),
        "dataset_scope": processed_dir.as_posix(),
        "files_checked": files_checked,
        "split_like_files": split_like_files,
        "total_records": total_records,
        "total_tokens": total_tokens,
        "mapping_summary": mapping_summary,
        "namespace_summary": dict(namespace_summary.most_common()),
        "edge_case_counts": dict(edge_case_counts),
        "top_unmapped": {
            category: dict(counter.most_common(20))
            for category, counter in top_unmapped.items()
        },
        "top_unknown_namespace_examples": dict(
            unknown_namespace_examples.most_common(MAX_EXAMPLES)
        ),
        "top_malformed_examples": dict(
            malformed_examples.most_common(MAX_EXAMPLES)
        ),
        "top_other_examples": dict(other_examples.most_common(MAX_EXAMPLES)),
        "top_duplicate_examples": dict(
            duplicate_examples.most_common(MAX_EXAMPLES)
        ),
        "critical_issues": critical_issues,
        "warnings": warnings,
        "milestone1_ready": milestone1_ready,
    }
    return report


def build_edge_case_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Extract a minimal summary restricted to edge cases, critical issues, and unmapped tokens."""
    return {
        "generated_at": report["generated_at"],
        "dataset_scope": report["dataset_scope"],
        "edge_case_counts": report["edge_case_counts"],
        "critical_issues": report["critical_issues"],
        "warnings": report["warnings"],
        "top_unmapped": report["top_unmapped"],
        "top_unknown_namespace_examples": report.get(
            "top_unknown_namespace_examples", {}
        ),
        "top_malformed_examples": report.get("top_malformed_examples", {}),
        "top_other_examples": report.get("top_other_examples", {}),
        "top_duplicate_examples": report.get("top_duplicate_examples", {}),
        "milestone1_ready": report["milestone1_ready"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the generated audit report payload into a human-readable Markdown format."""
    lines: list[str] = []
    lines.append("# Day 13 Mapping Audit")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Dataset scope: `{report['dataset_scope']}`")
    lines.append(f"- Total records: **{report['total_records']}**")
    lines.append(f"- Total tokens: **{report['total_tokens']}**")
    lines.append(f"- Milestone 1 ready: **{report['milestone1_ready']}**")
    lines.append("")

    lines.append("## Mapping Summary")
    lines.append("")
    lines.append("| Category | Total | Mapped | Unmapped | Mapping Rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for category, row in report["mapping_summary"].items():
        rate = row["mapping_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.4f}"
        lines.append(
            f"| {category} | {row['total']} | {row['mapped']} | {row['unmapped']} | {rate_text} |"
        )
    lines.append("")

    lines.append("## Namespace Summary")
    lines.append("")
    if report["namespace_summary"]:
        for namespace, count in report["namespace_summary"].items():
            lines.append(f"- `{namespace}`: {count}")
    else:
        lines.append("- No namespace observations found.")
    lines.append("")

    lines.append("## Edge Case Counts")
    lines.append("")
    if report["edge_case_counts"]:
        for key, count in sorted(report["edge_case_counts"].items()):
            lines.append(f"- `{key}`: {count}")
    else:
        lines.append("- No edge cases counted.")
    lines.append("")

    lines.append("## Edge Case Examples")
    lines.append("")

    def _render_example_block(title: str, payload: dict[str, int]) -> None:
        lines.append(f"### {title}")
        if payload:
            for token, count in payload.items():
                lines.append(f"- `{token}`: {count}")
        else:
            lines.append("- None")
        lines.append("")

    _render_example_block(
        "Top Unknown Namespace Examples",
        report.get("top_unknown_namespace_examples", {}),
    )
    _render_example_block(
        "Top Malformed Examples",
        report.get("top_malformed_examples", {}),
    )
    _render_example_block(
        "Top Other Examples",
        report.get("top_other_examples", {}),
    )
    _render_example_block(
        "Top Duplicate Examples",
        report.get("top_duplicate_examples", {}),
    )

    lines.append("## Top Unmapped Tokens")
    lines.append("")
    for category, items in report["top_unmapped"].items():
        lines.append(f"### {category}")
        if items:
            for token, count in items.items():
                lines.append(f"- `{token}`: {count}")
        else:
            lines.append("- None")
        lines.append("")

    lines.append("## Critical Issues")
    lines.append("")
    if report["critical_issues"]:
        for item in report["critical_issues"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if report["warnings"]:
        for item in report["warnings"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")