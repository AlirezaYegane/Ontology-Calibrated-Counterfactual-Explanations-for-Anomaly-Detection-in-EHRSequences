from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .index import OntologyIndex
from .types import ClinicalRecord, OntologyViolation


class OntologyRule(Protocol):
    rule_id: str

    def check(self, record: ClinicalRecord, index: OntologyIndex) -> list[OntologyViolation]:
        ...


@dataclass
class DemographicRule:
    rule_id: str
    sex_to_forbidden_codes: dict[str, set[str]]

    def check(self, record: ClinicalRecord, index: OntologyIndex) -> list[OntologyViolation]:
        if not record.sex:
            return []

        forbidden = self.sex_to_forbidden_codes.get(record.sex.upper(), set())
        violations: list[OntologyViolation] = []

        for code in record.codes:
            if code in forbidden:
                violations.append(
                    OntologyViolation(
                        rule_id=self.rule_id,
                        kind="demographic_mismatch",
                        message=f"Code '{index.get_term(code)}' is incompatible with recorded sex '{record.sex}'.",
                        codes=(code,),
                        severity=1.0,
                    )
                )

        return violations


@dataclass
class RequiredCodesRule:
    rule_id: str

    def check(self, record: ClinicalRecord, index: OntologyIndex) -> list[OntologyViolation]:
        code_set = set(record.codes)
        violations: list[OntologyViolation] = []

        for code in record.codes:
            required = index.required_diagnoses_for_code.get(code, [])
            if required and not any(req in code_set for req in required):
                expected_terms = ", ".join(index.get_term(item) for item in required)
                violations.append(
                    OntologyViolation(
                        rule_id=self.rule_id,
                        kind="missing_required_code",
                        message=(
                            f"Code '{index.get_term(code)}' appears without any required supporting code: "
                            f"{expected_terms}."
                        ),
                        codes=(code, *tuple(required)),
                        severity=1.0,
                    )
                )

        return violations


@dataclass
class MutualExclusionRule:
    rule_id: str

    def check(self, record: ClinicalRecord, index: OntologyIndex) -> list[OntologyViolation]:
        code_set = set(record.codes)
        violations: list[OntologyViolation] = []

        for left, right in sorted(index.mutually_exclusive_pairs):
            if left in code_set and right in code_set:
                violations.append(
                    OntologyViolation(
                        rule_id=self.rule_id,
                        kind="mutual_exclusion",
                        message=(
                            f"Codes '{index.get_term(left)}' and '{index.get_term(right)}' should not co-occur."
                        ),
                        codes=(left, right),
                        severity=1.0,
                    )
                )

        return violations
