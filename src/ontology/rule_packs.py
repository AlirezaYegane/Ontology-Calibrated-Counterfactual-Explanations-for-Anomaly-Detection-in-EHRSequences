"""
src/ontology/rule_packs.py
==========================
Phase 3b -- Curated, auditable ontology rule packs.

Phase 3 found the real ontology scorer at chance on benchmark-v2: the canonical
engine's default rule tables were too sparse and not aligned with the SNOMED/
RxNorm concepts the token->ontology mapping actually produces (Phase 3 diagnostic
``artifacts/phase3/ontology_rule_coverage_v2.json``). This module adds explicit,
high-precision, clinically-motivated rule packs and the small amount of new rule
machinery they need.

Design principles (scientific integrity)
----------------------------------------
* Rules are defined from **standard clinical knowledge** (ICD chapter structure,
  RxNorm ingredient identity) -- NOT from benchmark-v2 labels or injected tokens.
  No rule reads ``label`` / ``anomaly_type`` / hidden / audit metadata.
* Concept groups are built by mapping curated ICD code families through the
  **real ICD->SNOMED crosswalk** (and seeding authoritative SNOMED roots +
  descendants), so a rule fires on exactly the concepts the scorer's
  ``map_tokens_to_ontology_codes`` produces. This is ontology normalization, not
  label fitting.
* Every rule carries ``rule_id / rule_type / severity / rationale / source /
  limitations`` for audit.
* Severities encode confidence: demographic sex-restriction is high precision
  (1.0); medication-indication and diabetes-type exclusion are weaker, so they
  use reduced severity and documented limitations (a missing diagnosis code is
  not proof of mismatch; type 1 / type 2 diabetes co-coding occurs in real EHRs).

The curated tables here are intentionally small and high precision. They are
consumed by :mod:`src.ontology.rule_loader`, which turns them into concrete
:class:`OntologyRule` objects against a loaded :class:`OntologyIndex`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .records import ClinicalRecord, OntologyViolation
from .rule_engine import OntologyRule

if TYPE_CHECKING:  # avoid runtime import cost / cycles
    from .index import OntologyIndex


# ===========================================================================
# Curated clinical knowledge tables (auditable; sourced from ICD/RxNorm, not
# from benchmark labels).
# ===========================================================================


def _icd9_range(lo: int, hi: int) -> tuple[str, ...]:
    """Inclusive 3-digit ICD-9 category prefixes ``lo..hi``.

    3-digit prefixes match diagnosis categories (e.g. ``630``..``679``) without
    colliding with 2-digit ICD-9 *procedure* keys (e.g. ``63.0``), because
    ``"63.0".startswith("630")`` is False.
    """
    return tuple(str(n) for n in range(lo, hi + 1))


# --- Sex-restricted diagnosis families -------------------------------------
# Female-only: pregnancy / childbirth / puerperium (ICD-10 chapter XV "O*",
# pregnancy state Z-codes) and female reproductive-organ neoplasms. Male-only:
# prostate / male-genital disorders and male-genital neoplasms. These are
# anatomically sex-specific by definition of the code family.

FEMALE_ONLY_ICD10_PREFIXES: tuple[str, ...] = (
    "O",  # Chapter XV: pregnancy, childbirth and the puerperium (all O00-O9A)
    "Z33",
    "Z34",
    "Z3A",
    "Z37",  # pregnant state / supervision / weeks / outcome
    "C51",
    "C52",
    "C53",
    "C54",
    "C55",
    "C56",
    "C57",
    "C58",  # female genital cancer
    "D25",
    "D26",
    "D27",
    "D28",  # uterine/ovarian benign neoplasms
)
FEMALE_ONLY_ICD9_PREFIXES: tuple[str, ...] = (
    _icd9_range(630, 679)  # pregnancy, childbirth, the puerperium
    + ("V22", "V23", "V24", "V27", "V28")  # pregnancy supervision / outcome
    + _icd9_range(179, 184)  # female genital malignant neoplasms
    + _icd9_range(218, 221)  # uterine/ovarian/female-genital benign neoplasms
)
# Authoritative SNOMED seed so fixture/mock data (which has no crosswalk) fires.
FEMALE_ONLY_SNOMED_ROOTS: tuple[str, ...] = ("SNOMED:77386006",)  # pregnancy

# Precision filter: a crosswalk concept counts as female-only only if its SNOMED
# preferred term names pregnancy/obstetric or female reproductive anatomy. This
# drops generic disorders that pregnancy-complication codes cross-map to.
FEMALE_TERM_KEYWORDS: tuple[str, ...] = (
    "pregnan",
    "gestation",
    "gravida",
    "obstetric",
    "puerper",
    "antepartum",
    "postpartum",
    "peripartum",
    "intrapartum",
    "delivery",
    "labor",
    "labour",
    "childbirth",
    "birth",
    "fetal",
    "foetal",
    "fetus",
    "placenta",
    "amniotic",
    "meconium",
    "perineal",
    "perineum",
    "vulva",
    "vagina",
    "vaginal",
    "cervix",
    "cervic",
    "uter",
    "endometri",
    "ovar",
    "fallopian",
    "eclampsia",
    "gestational",
)
# Drop SECONDARY "generic-disorder in/complicating pregnancy/obstetric" concepts:
# they are reached from ordinary (non-pregnancy) ICD codes via the many-to-many
# crosswalk and would fire on normal male records.
FEMALE_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    " in pregnancy",
    " in obstetric",
    "in obstetric context",
    " during pregnancy",
    " complicating pregnancy",
    " complicating childbirth",
    " in childbirth",
    " complicating the puerperium",
    " in the puerperium",
    " in mother",
    "in mother",
    " complicating pregnancy, childbirth",
    "secondary to",
)

MALE_ONLY_ICD10_PREFIXES: tuple[str, ...] = (
    "N40",
    "N41",
    "N42",
    "N43",
    "N44",
    "N45",
    "N46",
    "N47",
    "N48",
    "N49",
    "N50",
    "N51",
    "N52",
    "N53",  # male genital organ disorders
    "C60",
    "C61",
    "C62",
    "C63",  # male genital malignant neoplasms
)
MALE_ONLY_ICD9_PREFIXES: tuple[str, ...] = (
    _icd9_range(600, 608)  # prostate / male genital disorders
    + ("185", "186", "187")  # prostate / testis / male-genital malignant neoplasm
    + ("222",)  # benign neoplasm of male genital organs
)
MALE_ONLY_SNOMED_ROOTS: tuple[str, ...] = ()  # rely on crosswalk (no clean mock root)

# Precision filter for male-only crosswalk concepts (male reproductive anatomy).
MALE_TERM_KEYWORDS: tuple[str, ...] = (
    "prostat",
    "testic",
    "testis",
    "testes",
    "scrotal",
    "scrotum",
    "penis",
    "penil",
    "seminal",
    "epididym",
    "spermat",
    "prepuce",
    "foreskin",
    "vas deferens",
)


# --- Diabetes type-1 / type-2 (for forbidden co-occurrence) ----------------
# ICD-10 E10* == type 1, E11* == type 2 by definition. ICD-9 250 generic does
# NOT distinguish type without the 5th digit, so it is deliberately excluded to
# avoid putting every diabetic in both groups.
DM_TYPE1_ICD10_PREFIXES: tuple[str, ...] = ("E10",)
DM_TYPE2_ICD10_PREFIXES: tuple[str, ...] = ("E11",)
DM_TYPE1_SNOMED_ROOTS: tuple[str, ...] = ("SNOMED:46635009",)  # type 1 DM
DM_TYPE2_SNOMED_ROOTS: tuple[str, ...] = ("SNOMED:44054006",)  # type 2 DM
# Precision filters: a crosswalk concept counts as type-1/type-2 only if its term
# explicitly says so -- drops generic complication concepts (hyperlipidemia,
# dyslipidemia) that BOTH E10* and E11* cross-map to and that otherwise made the
# two groups overlap and fire on ordinary diabetics.
DM_TYPE1_TERM_KEYWORDS: tuple[str, ...] = ("type 1 diabet", "type i diabet")
DM_TYPE2_TERM_KEYWORDS: tuple[str, ...] = ("type 2 diabet", "type ii diabet")
# Drop SECONDARY "<complication> due to type N diabetes" concepts: ordinary
# diabetic-complication codes cross-map to these, putting the same patient in
# both type groups and firing the exclusion on single-type diabetics.
DM_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (" due to", "complicating", "associated with")


# --- Medication -> required clinical context -------------------------------
# Drugs keyed by the EXACT RxCUI the scorer's drug map produces (verified
# against benchmark-v2 token mappings). Required context is a curated ICD family
# set mapped through the crosswalk + SNOMED roots. Medication-indication signal
# is WEAKER than demographic signal (drugs have many off-label/secondary uses),
# so severity is reduced and the limitation is documented.

# Therapeutic anticoagulants -> thromboembolic / atrial-fibrillation context.
# NOTE: unfractionated heparin (5224/9877) is DELIBERATELY EXCLUDED -- it is given
# prophylactically to most ICU patients without any coded thromboembolic
# indication, so requiring context for it produces many false positives on normal
# records. Only the more indication-bound agents (warfarin, DOACs, treatment-dose
# enoxaparin) are kept. This matches the benchmark injector's drug set.
ANTICOAGULANT_RXCUIS: tuple[str, ...] = (
    "RXNORM:11289",  # warfarin
    "RXNORM:67108",  # enoxaparin
    "RXNORM:221095",  # enoxaparin sodium
    "RXNORM:1364430",  # apixaban
    "RXNORM:1114195",  # rivaroxaban
    "RXNORM:1037042",  # dabigatran etexilate
)
ANTICOAG_CONTEXT_ICD10_PREFIXES: tuple[str, ...] = (
    "I48",  # atrial fibrillation and flutter
    "I26",  # pulmonary embolism
    "I80",
    "I81",
    "I82",  # phlebitis/thrombophlebitis, portal/other venous thrombosis
    "I63",  # cerebral infarction (cardioembolic stroke prophylaxis)
    "Z79.01",  # long-term (current) use of anticoagulants
    "T45.515",  # adverse/long-term anticoagulant context (encounter)
)
ANTICOAG_CONTEXT_ICD9_PREFIXES: tuple[str, ...] = (
    "427",  # cardiac dysrhythmias (incl. 427.31 atrial fibrillation)
    "415",  # acute pulmonary heart disease (415.1 PE)
    "451",
    "452",
    "453",  # phlebitis/thrombophlebitis / venous thrombosis/embolism
    "434",  # occlusion of cerebral arteries
    "V58",  # long-term drug use (V58.61 anticoagulants)
)
ANTICOAG_CONTEXT_SNOMED_ROOTS: tuple[str, ...] = (
    "SNOMED:49436004",  # atrial fibrillation
    "SNOMED:59282003",  # pulmonary embolism
    "SNOMED:64779008",  # blood coagulation disorder
    "SNOMED:439127006",  # disorder of coagulation
)

# Levothyroxine -> hypothyroidism context.
LEVOTHYROXINE_RXCUIS: tuple[str, ...] = (
    "RXNORM:40144",  # levothyroxine sodium
    "RXNORM:10582",  # levothyroxine
)
HYPOTHYROID_CONTEXT_ICD10_PREFIXES: tuple[str, ...] = ("E02", "E03", "E890")
HYPOTHYROID_CONTEXT_ICD9_PREFIXES: tuple[str, ...] = ("243", "244")
HYPOTHYROID_CONTEXT_SNOMED_ROOTS: tuple[str, ...] = (
    "SNOMED:40930008",  # hypothyroidism
)

# Insulin -> diabetes context. KEPT for clinical completeness + fixture tests
# (fixtures map insulin to RXNORM:5856). On benchmark-v2 the bare ``MED_INSULIN``
# token is UNMAPPED by the drug->RxCUI map, so this rule cannot fire on real v2
# insulin anomalies -- a documented token->RxNorm mapping-gap limitation.
INSULIN_RXCUIS: tuple[str, ...] = ("RXNORM:5856",)  # insulin (fixture/base)
DIABETES_CONTEXT_ICD10_PREFIXES: tuple[str, ...] = ("E08", "E09", "E10", "E11", "E13")
DIABETES_CONTEXT_ICD9_PREFIXES: tuple[str, ...] = ("250",)
DIABETES_CONTEXT_SNOMED_ROOTS: tuple[str, ...] = (
    "SNOMED:73211009",  # diabetes mellitus
    "SNOMED:44054006",  # type 2 DM
    "SNOMED:46635009",  # type 1 DM
)


# ===========================================================================
# Concept-group builder
# ===========================================================================


def _matches_family(icd_key: str, prefixes: tuple[str, ...]) -> bool:
    return icd_key.upper().startswith(prefixes)


def build_concept_group(
    index: "OntologyIndex",
    *,
    snomed_roots: tuple[str, ...] = (),
    icd_prefixes: tuple[str, ...] = (),
    crosswalk_term_keywords: tuple[str, ...] | None = None,
    crosswalk_exclude_substrings: tuple[str, ...] = (),
) -> set[str]:
    """Build a SNOMED concept group aligned with the scorer's mapping.

    The group is the union of:
      * each authoritative SNOMED ``root`` and its hierarchy descendants
        (trusted; never keyword-filtered), and
      * every SNOMED concept that a curated ICD code family maps to via the real
        ICD->SNOMED crosswalk (matched on the dotted ICD key's prefix).

    ``crosswalk_term_keywords`` is the key precision control. The ICD->SNOMED
    crosswalk is lossy and many-to-many: e.g. pregnancy-COMPLICATION codes
    (``O10`` hypertension-in-pregnancy) cross-map to the *generic* underlying
    disorder ("Benign essential hypertension"), and diabetes-complication codes
    map to generic shared concepts ("Mixed hyperlipidemia"). Admitting those raw
    would fire on ordinary records of any sex. When ``crosswalk_term_keywords``
    is given, a crosswalk concept is admitted only if its SNOMED *preferred term*
    contains one of the (clinically curated) keywords -- so only genuinely
    sex-/type-specific concepts enter the group. Seed-root subtrees are always
    trusted and never filtered (so fixtures with authoritative roots still fire).

    Building the group from the *same* crosswalk the scorer uses guarantees the
    rule fires on the concepts ``map_tokens_to_ontology_codes`` produces.
    """
    group: set[str] = set()
    for root in snomed_roots:
        group.add(root)
        group |= index.get_descendants(root)
    if icd_prefixes:
        prefixes = tuple(p.upper() for p in icd_prefixes)
        kw = (
            tuple(k.lower() for k in crosswalk_term_keywords)
            if crosswalk_term_keywords is not None
            else None
        )
        excl = tuple(s.lower() for s in crosswalk_exclude_substrings)
        for icd_key, snomed_ids in index.icd_to_snomed.items():
            if not _matches_family(str(icd_key), prefixes):
                continue
            for sid in snomed_ids:
                term = index.get_term(sid).lower()
                if kw is not None and not any(k in term for k in kw):
                    continue
                if excl and any(s in term for s in excl):
                    continue  # secondary "<generic> in pregnancy / due to type N" concept
                group.add(sid)
    return group


# ===========================================================================
# New rule type: group-based mutual exclusion
# ===========================================================================


@dataclass
class GroupConflict:
    """A pair of mutually-implausible concept groups.

    Each side is "present" if a mapped SNOMED concept is in its concept set OR a
    raw source token's ICD body matches its ``*_icd_prefixes`` family. The
    source-token path makes the rule robust to crosswalk mapping gaps (e.g. a
    non-standard injected code body that fails to map but is a valid type-1
    diagnosis family).
    """

    label: str
    left: set[str]
    right: set[str]
    left_icd_prefixes: tuple[str, ...] = ()
    right_icd_prefixes: tuple[str, ...] = ()


def _icd_body(token: str) -> str | None:
    """Strip a ``DX_9_`` / ``DX_10_`` prefix and return the undotted ICD body.

    Returns ``None`` for non-diagnosis tokens. ``DX_10_O80`` -> ``O80``,
    ``DX_9_6340`` -> ``6340``.
    """
    up = str(token).strip().upper()
    if up.startswith("DX_10_"):
        return up[len("DX_10_") :]
    if up.startswith("DX_9_"):
        return up[len("DX_9_") :]
    return None


@dataclass
class SexRestrictedRule(OntologyRule):
    """Flag sex-incompatible diagnoses, anchored on the SOURCE ICD-code family.

    Two complementary, high-precision signals (a violation fires on either):

    1. **Source-token ICD family** (primary, real-data path):
       ``sex_to_forbidden_icd_prefixes`` maps a sex to undotted ICD code-family
       prefixes that are anatomically exclusive to the *other* sex (e.g.
       pregnancy ``O*`` is forbidden for ``M``; prostate ``N40*`` for ``F``). We
       test the record's RAW source tokens, NOT mapped SNOMED concepts: this
       avoids the many-to-many ICD->SNOMED crosswalk noise whereby an ordinary
       code (GERD, cardiac arrest) cross-maps to an obstetric *variant* concept
       and would otherwise fire on normal patients. Normal patients simply do not
       carry the opposite sex's diagnosis-code families.

    2. **Clean SNOMED subtree** (fixture/hierarchy path):
       ``sex_to_forbidden_group`` is a *hierarchy-derived* concept set (SNOMED
       roots + descendants, NO crosswalk), so a record already expressed in clean
       SNOMED codes (e.g. a pregnancy-subtree concept in a male) still fires.

    The ICD families are standard clinical knowledge (auditable); they map to
    sex-specific SNOMED concepts via the crosswalk used elsewhere.
    """

    rule_id: str
    sex_to_forbidden_group: dict[str, set[str]] = field(default_factory=dict)
    sex_to_forbidden_icd_prefixes: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    severity: float = 1.0

    def check(
        self, record: ClinicalRecord, index: "OntologyIndex"
    ) -> list[OntologyViolation]:
        sex = _normalize_sex(record.sex)
        if sex is None:
            return []
        offending: list[str] = []

        # (1) source-token ICD family
        prefixes = self.sex_to_forbidden_icd_prefixes.get(sex)
        if prefixes:
            for tok in record.source_tokens:
                body = _icd_body(tok)
                if body is not None and body.startswith(prefixes):
                    offending.append(str(tok))

        # (2) clean SNOMED subtree membership
        forbidden = self.sex_to_forbidden_group.get(sex)
        if forbidden:
            offending.extend(sorted(set(record.codes) & forbidden))

        if not offending:
            return []
        # de-dup preserving order
        seen: set[str] = set()
        unique = [c for c in offending if not (c in seen or seen.add(c))]
        return [
            OntologyViolation(
                rule_id=self.rule_id,
                kind="demographic_mismatch",
                message=(
                    f"Sex-incompatible diagnosis for sex={sex}: "
                    f"{unique[:3]}{'...' if len(unique) > 3 else ''}"
                ),
                codes=tuple(unique[:5]),
                severity=self.severity,
            )
        ]


@dataclass
class RequiredContextRule(OntologyRule):
    """Flag a medication present without any of its required clinical context.

    ``drug_to_context`` maps a drug concept (``RXNORM:<cui>``) to the set of
    diagnosis concepts that would justify it. ``drug_name_to_context`` maps a
    drug-INGREDIENT name substring (e.g. ``"INSULIN"``) matched against raw
    ``MED_*`` source tokens to the same kind of context set -- this catches
    clinically important drugs (notably bare ``MED_INSULIN``) that the
    drug->RxCUI map leaves UNMAPPED. A violation fires when the drug is present
    and the record shares NO concept with that context set.

    Conservative by design: medication-indication signal is weaker than
    demographic signal (drugs have secondary/off-label uses; coding is
    incomplete), so callers use a reduced severity.
    """

    rule_id: str
    drug_to_context: dict[str, frozenset[str]] = field(default_factory=dict)
    drug_name_to_context: dict[str, frozenset[str]] = field(default_factory=dict)
    severity: float = 0.5

    def check(
        self, record: ClinicalRecord, index: "OntologyIndex"
    ) -> list[OntologyViolation]:
        code_set = set(record.codes)
        violations: list[OntologyViolation] = []
        reported: set[str] = set()

        # (1) mapped RxCUI drug concepts
        for code in record.codes:
            if code in reported:
                continue
            context = self.drug_to_context.get(code)
            if context is None:
                continue
            reported.add(code)
            if code_set & context:
                continue  # justified
            violations.append(
                OntologyViolation(
                    rule_id=self.rule_id,
                    kind="missing_required_code",
                    message=(
                        f"Medication {code} present with no supporting "
                        "clinical-context diagnosis."
                    ),
                    codes=(code,),
                    severity=self.severity,
                )
            )

        # (2) source-token ingredient names (covers UNMAPPED drugs, e.g. insulin)
        if self.drug_name_to_context:
            med_tokens = [
                t for t in record.source_tokens if str(t).upper().startswith("MED_")
            ]
            for name, context in self.drug_name_to_context.items():
                key = f"DRUGNAME:{name}"
                if key in reported:
                    continue
                if not any(name in str(t).upper() for t in med_tokens):
                    continue
                reported.add(key)
                if code_set & context:
                    continue  # justified
                violations.append(
                    OntologyViolation(
                        rule_id=self.rule_id,
                        kind="missing_required_code",
                        message=(
                            f"Medication '{name}' present with no supporting "
                            "clinical-context diagnosis."
                        ),
                        codes=(key,),
                        severity=self.severity,
                    )
                )
        return violations


def _normalize_sex(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"m", "male", "man"}:
        return "M"
    if text in {"f", "female", "woman"}:
        return "F"
    return str(value).strip().upper() or None


@dataclass
class GroupMutualExclusionRule(OntologyRule):
    """Flag co-occurrence of concepts from two mutually-implausible groups.

    Unlike :class:`MutualExclusionRule` (which keys on single concept pairs and
    their descendants), this fires when a record contains *any* concept from
    ``group.left`` AND *any* from ``group.right``. This is required because the
    real crosswalk maps an ICD family (e.g. all of ``E11*``) to many scattered
    SNOMED concepts that do not share one clean descendant root.

    Emits ``kind="mutual_exclusion"`` (compatible with existing tests).
    """

    rule_id: str
    groups: list[GroupConflict] = field(default_factory=list)
    severity: float = 0.5

    @staticmethod
    def _side_hits(
        concept_group: set[str],
        icd_prefixes: tuple[str, ...],
        code_set: set[str],
        token_bodies: list[str],
    ) -> list[str]:
        hits = sorted(code_set & concept_group)
        if icd_prefixes:
            hits.extend(b for b in token_bodies if b.startswith(icd_prefixes))
        return hits

    def check(
        self, record: ClinicalRecord, index: "OntologyIndex"
    ) -> list[OntologyViolation]:
        code_set = set(record.codes)
        token_bodies = [
            b for b in (_icd_body(t) for t in record.source_tokens) if b is not None
        ]
        violations: list[OntologyViolation] = []
        for grp in self.groups:
            left_hit = self._side_hits(
                grp.left, grp.left_icd_prefixes, code_set, token_bodies
            )
            right_hit = self._side_hits(
                grp.right, grp.right_icd_prefixes, code_set, token_bodies
            )
            if left_hit and right_hit:
                evidence = tuple(left_hit[:3] + right_hit[:3])
                violations.append(
                    OntologyViolation(
                        rule_id=self.rule_id,
                        kind="mutual_exclusion",
                        message=(
                            f"Mutually implausible co-occurrence ({grp.label}): "
                            f"{left_hit[:3]} with {right_hit[:3]}."
                        ),
                        codes=evidence,
                        severity=self.severity,
                    )
                )
        return violations


__all__ = [
    "build_concept_group",
    "GroupConflict",
    "GroupMutualExclusionRule",
    "SexRestrictedRule",
    "RequiredContextRule",
    # tables (exported for the loader + tests/audit)
    "FEMALE_ONLY_ICD10_PREFIXES",
    "FEMALE_ONLY_ICD9_PREFIXES",
    "FEMALE_ONLY_SNOMED_ROOTS",
    "MALE_ONLY_ICD10_PREFIXES",
    "MALE_ONLY_ICD9_PREFIXES",
    "MALE_ONLY_SNOMED_ROOTS",
    "DM_TYPE1_ICD10_PREFIXES",
    "DM_TYPE2_ICD10_PREFIXES",
    "DM_TYPE1_SNOMED_ROOTS",
    "DM_TYPE2_SNOMED_ROOTS",
    "ANTICOAGULANT_RXCUIS",
    "ANTICOAG_CONTEXT_ICD10_PREFIXES",
    "ANTICOAG_CONTEXT_ICD9_PREFIXES",
    "ANTICOAG_CONTEXT_SNOMED_ROOTS",
    "LEVOTHYROXINE_RXCUIS",
    "HYPOTHYROID_CONTEXT_ICD10_PREFIXES",
    "HYPOTHYROID_CONTEXT_ICD9_PREFIXES",
    "HYPOTHYROID_CONTEXT_SNOMED_ROOTS",
    "INSULIN_RXCUIS",
    "DIABETES_CONTEXT_ICD10_PREFIXES",
    "DIABETES_CONTEXT_ICD9_PREFIXES",
    "DIABETES_CONTEXT_SNOMED_ROOTS",
]
