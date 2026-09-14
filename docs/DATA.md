# Data and representations

The repository does not redistribute benchmark datasets or trained encoder/model
weights. Users must obtain each source under its original terms.

## Benchmarks

Paper configuration templates are supplied for DrugBank, Davis, KIBA, Yamanishi
08, Hetionet, Activation, and Inhibition. Pairwise-random evaluation is used for
the EviDTI-style DrugBank/Davis/KIBA experiments. Entity-aware benchmarks use
warm-start, compound-disjoint, and target-disjoint regimes.

## Feature NPZ format

The lightweight release trainer accepts an `.npz` file containing:

- `smiles`: `[N, D_s]` pooled MoLFormer vectors;
- `text`: `[N, D_t]` pooled MolT5 vectors;
- `protein`: `[N, D_p]` pooled ESM-2 vectors;
- `labels`: `[N]` binary interaction labels;
- `split`: `[N]`, where `0=train`, `1=validation`, `2=test`.

Optional relational fields are:

- `graph_context`: `[N, D_g]`;
- `graph_available`: `[N]`;
- `kge_score`: `[N]`;
- `kge_available`: `[N]`;
- `has_text`: `[N]`.

The built-in `make-synthetic` command produces this format for smoke testing.

## Chemical preprocessing

The manuscript preprocessing parses structures with RDKit, applies
`MolStandardize.Cleanup`, retains the principal component with `FragmentParent`,
normalizes formal charge with `Uncharger`, and exports canonical isomeric SMILES.
The clean release intentionally does not fabricate missing benchmark-specific
mapping rules; dataset-specific identifier resolution remains the responsibility
of the corresponding data-preparation pipeline.

## Strict Hetionet graph control

For the strict Hetionet analysis, held-out validation/test direct compound-target
edges and their generated inverse edges are removed before fold-specific KGE
training. Held-out positive pairs are excluded from KGE negative sampling. Other
biomedical relations involving the same entities may remain, so this protocol
must not be described as removing every possible graph path.

