from __future__ import annotations

from scrdti.preprocessing import (
    assert_regime,
    audit_pair_splits,
    canonicalize_smiles,
    normalize_protein_sequence,
)


def test_manuscript_smiles_standardization_keeps_principal_component() -> None:
    # Sodium chloride is a disconnected counterion; FragmentParent retains ethanol.
    standardized = canonicalize_smiles("CCO.[Na+].[Cl-]")
    assert standardized == "CCO"


def test_protein_sequence_normalization() -> None:
    assert normalize_protein_sequence("acD ef\nG") == "ACDEFG"


def test_split_audit_enforces_entity_disjoint_regimes() -> None:
    train = [("d1", "p1"), ("d2", "p2")]
    compound_cold = [("d3", "p1")]
    audit = audit_pair_splits(train, compound_cold)
    assert audit.compound_overlap == 0
    assert audit.target_overlap == 1
    assert_regime(audit, "compound-disjoint")

