# Reproducibility

## What this repository guarantees

The release is designed to provide a clean, paper-facing implementation that can
be installed and exercised without private cluster paths or bundled weights. CI/
local tests cover model construction, synthetic forward passes, evidence masks,
confidence-margin behavior, bounded KGE refinement, config parsing, historical checkpoint
mapping, and a tiny CPU train-predict-evaluate flow.

## Reproduction checklist

1. Record the Git commit, Python version, PyTorch build, operating system, and
   CPU/CUDA device.
2. Download source data from `docs/DATA.md`, retaining licenses and checksums.
3. Convert each benchmark to one pair-level CSV with `smiles`,
   `protein_sequence`, `label`, and `split`.
4. Freeze the split before `scrdti precompute`; use `0/1/2` for
   train/validation/test and never fit calibration on test predictions.
5. Run `precompute`, `train`, `predict`, and `evaluate` with matching YAML
   dimensions.
6. For graph/KGE experiments, save fold-specific triples, forbidden held-out
   pairs, KGE seed, and optional NPZ arrays with the checkpoint.
7. Select checkpoints by validation AUPRC and report test AUPRC as the primary
   ranking metric; report AUROC and calibration metrics separately.

Exact manuscript reproduction still requires dataset-specific identifier
mapping and fold preparation because the original sources use different
formats, licenses, and split definitions.

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
