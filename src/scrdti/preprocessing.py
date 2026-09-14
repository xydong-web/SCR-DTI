from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def canonicalize_smiles(smiles: str) -> str:
    """Apply the manuscript's RDKit structure-standardization steps.

    Steps: parse -> MolStandardize.Cleanup -> FragmentParent -> Uncharger ->
    canonical isomeric SMILES. Tautomer canonicalization, pH-dependent protomer
    enumeration, and 3-D conformer generation are intentionally not performed.
    """

    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SMILES standardization requires `pip install -e '.[chem]'`."
        ) from exc
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def normalize_protein_sequence(sequence: str) -> str:
    sequence = "".join(str(sequence).split()).upper()
    if not sequence:
        raise ValueError("protein sequence is empty")
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        raise ValueError(f"protein sequence contains unsupported symbols: {invalid}")
    return sequence


@dataclass(frozen=True)
class SplitAudit:
    pair_overlap: int
    compound_overlap: int
    target_overlap: int


def audit_pair_splits(
    train_pairs: Iterable[tuple[str, str]],
    heldout_pairs: Iterable[tuple[str, str]],
) -> SplitAudit:
    train = {(str(compound), str(target)) for compound, target in train_pairs}
    heldout = {(str(compound), str(target)) for compound, target in heldout_pairs}
    train_compounds = {compound for compound, _ in train}
    heldout_compounds = {compound for compound, _ in heldout}
    train_targets = {target for _, target in train}
    heldout_targets = {target for _, target in heldout}
    return SplitAudit(
        pair_overlap=len(train & heldout),
        compound_overlap=len(train_compounds & heldout_compounds),
        target_overlap=len(train_targets & heldout_targets),
    )


def assert_regime(
    audit: SplitAudit,
    regime: str,
) -> None:
    normalized = regime.strip().lower().replace("_", "-")
    if audit.pair_overlap:
        raise ValueError(f"pair leakage detected: {audit.pair_overlap} overlapping pairs")
    if normalized == "compound-disjoint" and audit.compound_overlap:
        raise ValueError(
            f"compound-disjoint violation: {audit.compound_overlap} compounds overlap"
        )
    if normalized == "target-disjoint" and audit.target_overlap:
        raise ValueError(f"target-disjoint violation: {audit.target_overlap} targets overlap")
    if normalized not in {"pairwise-random", "warm-start", "compound-disjoint", "target-disjoint"}:
        raise ValueError(f"unsupported evaluation regime: {regime}")

