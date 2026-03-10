from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ClinicalRecord:
    record_id: str
    codes: tuple[str, ...]
    sex: Optional[str] = None
    age_group: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OntologyViolation:
    rule_id: str
    kind: str
    message: str
    codes: tuple[str, ...] = ()
    severity: float = 1.0
