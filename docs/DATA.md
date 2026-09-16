# Data and representations

The repository does not redistribute benchmark datasets or trained encoder/model
weights. Users must obtain each source under its original terms.

## Where to download inputs

Use the following source pages and check their current licenses before use:

| Input | Source | Required material |
|---|---|---|
| DrugBank | [DrugBank releases](https://go.drugbank.com/releases/latest) | licensed compound-target pairs, structures, and target sequences |
| Davis, KIBA, Yamanishi 08 | [DeepDTA data](https://github.com/hkmztrk/DeepDTA/tree/master/data) | pair labels, SMILES, protein sequences, and the declared split |
| Hetionet | [Hetionet repository](https://github.com/hetio/hetionet) | relations and compound-target edges |
| Activation, Inhibition | The supplementary/source data for the corresponding benchmark publication | pair labels, structures, sequences, and splits |
| MoLFormer | [Hugging Face](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct) | canonical-SMILES embeddings |
| MolT5 | [Hugging Face](https://huggingface.co/laituan245/molt5-base) | `SMILES=` molecular-text embeddings |
| ESM-2 | [Hugging Face](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | protein-sequence embeddings |

`scrdti precompute` downloads the three Hugging Face models automatically.
Benchmark downloads are intentionally manual because DrugBank is licensed data
and the other benchmarks use different identifiers and redistribution terms.
Keep raw files outside Git and record source version, checksum, and license.

Recommended layout:

```text
data/raw/                 # downloaded source files; do not commit
data/prepared/pairs.csv   # pair-level input to precompute
data/prepared/pairs.npz   # generated pooled features
runs/                     # checkpoints and predictions; do not commit
```

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

`scrdti precompute` writes `smiles`, `text`, `protein`, `labels`, `split`, and
`has_text`. It does not generate `graph_context`, `kge_score`, or their masks;
those optional arrays must come from the graph/KGE preparation pipeline.

## CSV preparation contract

The CLI expects one row per compound-target pair, with `smiles`,
`protein_sequence`, `label`, and `split` columns. Labels are binary. For
compound-disjoint or target-disjoint evaluation, freeze the split before
precomputation and ensure held-out entities do not occur in training. Preserve
source identifiers in extra columns so predictions can be joined back later.

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
