from __future__ import annotations

import pandas as pd

from scrdti.graph_control import assert_no_heldout_direct_edges, remove_heldout_direct_edges
from scrdti.kge import score_pairs, train_toruse


def test_direct_and_inverse_heldout_edges_are_removed() -> None:
    triples = pd.DataFrame(
        [
            ("drugA", "drug_protein", "targetA"),
            ("targetA", "drug_protein_inverse", "drugA"),
            ("drugA", "drug_disease", "diseaseX"),
            ("drugB", "drug_protein", "targetB"),
        ],
        columns=["head", "relation", "tail"],
    )
    heldout = {("drugA", "targetA")}
    cleaned = remove_heldout_direct_edges(
        triples,
        heldout,
        forward_relations={"drug_protein"},
        inverse_relations={"drug_protein_inverse"},
    )
    assert len(cleaned) == 2
    assert "drug_disease" in set(cleaned["relation"])
    assert_no_heldout_direct_edges(
        cleaned,
        heldout,
        forward_relations={"drug_protein"},
        inverse_relations={"drug_protein_inverse"},
    )


def test_toruse_tiny_training_and_scoring() -> None:
    triples = pd.DataFrame(
        [
            ("drugA", "interacts", "targetA"),
            ("drugB", "interacts", "targetB"),
            ("drugA", "related", "drugB"),
            ("targetA", "related", "targetB"),
        ],
        columns=["head", "relation", "tail"],
    )
    model, index = train_toruse(
        triples,
        embedding_dim=8,
        epochs=1,
        batch_size=4,
        negative_ratio=1,
        seed=4,
        forbidden_pairs={("drugC", "targetC")},
    )
    scores = score_pairs(
        model,
        index,
        [("drugA", "targetA"), ("unknown", "targetA")],
        relation="interacts",
    )
    assert len(scores) == 2
    assert scores[0] == scores[0]
    assert scores[1] != scores[1]

