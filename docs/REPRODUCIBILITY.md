# Reproducibility

## What this repository guarantees

The release is designed to provide a clean, paper-facing implementation that can
be installed and exercised without private cluster paths or bundled weights. CI/
local tests cover model construction, synthetic forward passes, evidence masks,
confidence-margin behavior, bounded KGE refinement, config parsing, historical checkpoint
mapping, and a tiny CPU train-predict-evaluate flow.

## Basic workflow

Download the source data listed in `docs/DATA.md`, convert each benchmark to a
pair-level CSV, and run `scrdti precompute` followed by `scrdti train`,
`scrdti predict`, and `scrdti evaluate` with the matching YAML configuration.
For graph/KGE experiments, prepare the fold-specific graph artifacts before
training and include the optional arrays described in `docs/DATA.md`.

## What is intentionally external

- pretrained MoLFormer, MolT5, and ESM-2 weights;
- trained SCR-DTI checkpoints;
- DrugBank and other benchmark source data;
- precomputed embeddings and graph/KGE artifacts;
- historical exploratory sweeps and private cluster launchers.

## Historical paper checkpoints

Historical paper checkpoints contain 206 state-dict keys. Current development
files in the source research tree later gained 12 parameters under
`paper_binary_interaction_trunk` and `paper_binary_private_pair_adapter`. Those
parameters did not exist in the saved paper checkpoints. The release compatibility
loader therefore reconstructs the checkpoint-era prediction path and maps only
paper-relevant weights into `SCRDTI`.

The ignored legacy scaffolding includes condition encoders, latent/MoE plumbing,
auxiliary affinity/structure heads, the pretraining IC50 classifier, and a graph
feature adapter that did not contribute to the released paper-model logit path. The
loader reports ignored prefixes instead of silently treating them as public API.

## Evaluation

AUPRC is the primary ranking metric and AUROC is complementary. For probability
quality, the release includes validation-only temperature scaling plus Brier,
NLL, and expected calibration error helpers. Calibration parameters must never be
fit on test predictions.

The strict graph-control helpers are provided in `scrdti.graph_control`, and the
fold-specific TorusE implementation is in `scrdti.kge`. The default strict KGE
settings documented by the manuscript are 256-dimensional embeddings, 5 epochs,
batch size 1,024, Adam with learning rate 1e-3, head/tail corruption, a negative
ratio of 1, and seed 42. Held-out validation/test DTI pairs should be supplied to
the KGE negative sampler's `forbidden_pairs` set in addition to removing their
direct/inverse triples before training.

## Environment

The source project core tests were verified with Python 3.10.14 and the versions
listed in `requirements-lock.txt`. PyTorch CPU and CUDA wheel identifiers differ;
install the wheel appropriate for the local platform. The release itself has no
CUDA-only code path.
