from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


Pair = tuple[str, str]


def _as_pair_set(pairs: Iterable[Pair]) -> set[Pair]:
    return {(str(compound), str(target)) for compound, target in pairs}


def remove_heldout_direct_edges(
    triples: pd.DataFrame,
    heldout_pairs: Iterable[Pair],
    *,
    forward_relations: Iterable[str],
    inverse_relations: Iterable[str],
    head_column: str = "head",
    relation_column: str = "relation",
    tail_column: str = "tail",
) -> pd.DataFrame:
    """Remove direct held-out DTI edges and explicitly named inverse relations.

    This function controls only direct exposure. It deliberately retains other
    biomedical relations involving the same entities, matching the manuscript's
    strict-control boundary.
    """

    required = {head_column, relation_column, tail_column}
    missing = required.difference(triples.columns)
    if missing:
        raise ValueError(f"triples are missing columns: {sorted(missing)}")
    heldout = _as_pair_set(heldout_pairs)
    forward = {str(value) for value in forward_relations}
    inverse = {str(value) for value in inverse_relations}
    keep: list[bool] = []
    for row in triples[[head_column, relation_column, tail_column]].itertuples(index=False):
        head, relation, tail = str(row[0]), str(row[1]), str(row[2])
        is_forward = relation in forward and (head, tail) in heldout
        is_inverse = relation in inverse and (tail, head) in heldout
        keep.append(not (is_forward or is_inverse))
    return triples.loc[keep].reset_index(drop=True)


def residual_direct_exposure(
    triples: pd.DataFrame,
    heldout_pairs: Iterable[Pair],
    *,
    forward_relations: Iterable[str],
    inverse_relations: Iterable[str],
    head_column: str = "head",
    relation_column: str = "relation",
    tail_column: str = "tail",
) -> list[tuple[str, str, str]]:
    heldout = _as_pair_set(heldout_pairs)
    forward = {str(value) for value in forward_relations}
    inverse = {str(value) for value in inverse_relations}
    residual: list[tuple[str, str, str]] = []
    for row in triples[[head_column, relation_column, tail_column]].itertuples(index=False):
        head, relation, tail = str(row[0]), str(row[1]), str(row[2])
        if relation in forward and (head, tail) in heldout:
            residual.append((head, relation, tail))
        elif relation in inverse and (tail, head) in heldout:
            residual.append((head, relation, tail))
    return residual


def assert_no_heldout_direct_edges(
    triples: pd.DataFrame,
    heldout_pairs: Iterable[Pair],
    *,
    forward_relations: Iterable[str],
    inverse_relations: Iterable[str],
    head_column: str = "head",
    relation_column: str = "relation",
    tail_column: str = "tail",
) -> None:
    residual = residual_direct_exposure(
        triples,
        heldout_pairs,
        forward_relations=forward_relations,
        inverse_relations=inverse_relations,
        head_column=head_column,
        relation_column=relation_column,
        tail_column=tail_column,
    )
    if residual:
        raise ValueError(
            f"direct-edge control failed: {len(residual)} held-out direct/inverse edges remain; "
            f"examples={residual[:5]}"
        )

