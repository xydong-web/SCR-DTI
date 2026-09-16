# SCR-DTI

**Asymmetric Evidence Factorization for Reliable Drug-Target Prediction Across Data Regimes**

SCR-DTI is the paper-facing implementation of a drug-target interaction (DTI)
model that assigns heterogeneous evidence different decision roles instead of
treating all sources as interchangeable fusion inputs.

The released model has three explicit stages:

1. **Pair-semantic interaction evidence.** MoLFormer molecular embeddings,
   MolT5 molecular-text embeddings, and ESM-2 protein embeddings are projected
   into a shared space. Hierarchical cross-attention (HCA) forms the primary
   pair-level interaction evidence.
2. **Independent graph-context support.** When graph context is available, it
   contributes a separate additive support logit to the base decision.
3. **Confidence-margin bounded KGE refinement.** A KGE-derived score can modify
   the base decision through a residual whose magnitude is controlled by the
   absolute base-logit margin and bounded by a scaled `tanh`.

This repository intentionally contains **no trained weights and no benchmark
data**. It also excludes historical architecture sweeps, MoE/prototype research
branches, docking/MD workflows, cluster-specific Slurm files, caches, and private
server paths.

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For raw-sequence embedding extraction, install the optional encoder stack:

```bash
python -m pip install -e ".[encoders,chem,dev]"
```

The verified source environment used Python 3.10.14, PyTorch 2.3.1+cu121,
Transformers 4.57.3, RDKit 2025.3.5, NumPy 1.26.4, pandas 2.0.0, SciPy 1.15.3,
scikit-learn 1.7.1, PyArrow 22.0.0, and PyYAML 6.0.2. CPU-only PyTorch is fully
supported by the release tests.

## Quick CPU smoke test

Generate a tiny feature dataset, train, predict, and evaluate without downloading
any pretrained model or benchmark data:

```bash
scrdti make-synthetic --output /tmp/scrdti_demo.npz --n 96 --seed 7
scrdti train --config configs/examples/tiny_cpu.yaml \
  --data /tmp/scrdti_demo.npz --output /tmp/scrdti_demo.pt
scrdti predict --config configs/examples/tiny_cpu.yaml \
  --data /tmp/scrdti_demo.npz --checkpoint /tmp/scrdti_demo.pt \
  --output /tmp/scrdti_predictions.csv
scrdti evaluate --predictions /tmp/scrdti_predictions.csv
```

## Paper-scale inputs

SCR-DTI consumes **offline pooled embeddings**:

- canonical SMILES: `ibm-research/MoLFormer-XL-both-10pct`;
- `SMILES=`-prefixed molecular text: `laituan245/molt5-base`;
- protein sequence: `facebook/esm2_t33_650M_UR50D`.

The paper uses attention-mask-aware mean pooling. Molecular inputs are limited to
256 tokens and protein inputs to 1,024 residues. Encoder weights are not bundled.
See `docs/DATA.md` and `scrdti precompute --help`.

## Paper configurations

`configs/paper/` documents the seven benchmark families used by the manuscript:
DrugBank, Davis, KIBA, Yamanishi 08, Hetionet, Activation, and Inhibition. The
entity-aware configs record warm-start, compound-disjoint, and target-disjoint
regimes. They are release templates: paths must be supplied by the user because
the underlying datasets are not redistributed.

The strict Hetionet protocol removes held-out validation/test **direct DTI edges
and their generated inverse edges** before fold-specific KGE training and blocks
the same pairs from KGE negative sampling. This controls direct-edge exposure; it
does not claim that every possible multi-hop graph leakage path is absent.

## Legacy checkpoint compatibility

Historical paper training artifacts used a 206-key model state. Two modules later
appeared in development YAML files but were absent at training time:
`paper_binary_interaction_trunk` and `paper_binary_private_pair_adapter`.

The clean public model does not expose those post-training drift modules. Use:

```python
from scrdti import SCRDTI, ModelConfig
from scrdti.compat import load_legacy_checkpoint

model = SCRDTI(ModelConfig.historical_checkpoint())
report = load_legacy_checkpoint(model, "legacy_multitask_best.pt")
print(report)
```

The compatibility layer maps only the modules that contribute to the released
paper-facing SCR-DTI prediction path and reports ignored historical scaffolding.
It never rewrites the source checkpoint.

## Reproducibility boundaries

- Validation AUPRC is used for checkpoint selection in the paper workflow.
- Weighted binary cross-entropy uses the training-partition class ratio by
  default (capped at 12 unless overridden).
- KGE standardization statistics are estimated from available training-partition
  KGE scores and stored in the checkpoint state.
- Validation-only temperature scaling is provided in `scrdti.calibration`.
- AUPRC is the primary ranking endpoint; AUROC is complementary.
- DrugBank and other data must be obtained under their original licenses.
- No bundled weight means cloning this repository alone does not reproduce the
  manuscript numbers; users must train models or supply a compatible checkpoint.

See `docs/ARCHITECTURE.md`, `docs/DATA.md`, and `docs/REPRODUCIBILITY.md`.

## Reproduce the release workflow

The commands below cover the complete release workflow: environment setup, a
CPU smoke test, feature precomputation, training, prediction, and evaluation.

### 1. Install

```bash
git clone https://github.com/xydong-web/SCR-DTI.git
cd SCR-DTI
python -m venv .venv
source .venv/bin/activate                 # PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[encoders,chem,dev]"
```

### 2. Run the verified CPU smoke test

```bash
scrdti make-synthetic --output runs/synthetic.npz --n 96 --seed 7
scrdti train --config configs/examples/tiny_cpu.yaml \
  --data runs/synthetic.npz --output runs/synthetic.pt --device cpu
scrdti predict --config configs/examples/tiny_cpu.yaml \
  --data runs/synthetic.npz --checkpoint runs/synthetic.pt \
  --output runs/synthetic_predictions.csv --split 2 --device cpu
scrdti evaluate --predictions runs/synthetic_predictions.csv
```

This is an installation smoke test, not a paper benchmark.

### 3. Prepare and run a paper-scale dataset

Create one pair-level CSV with `smiles`, `protein_sequence`, `label`, and
`split`. Use `split=0` for train, `1` for validation, and `2` for test. Then:

```bash
scrdti precompute --input data/pairs.csv --output data/pairs.npz \
  --smiles-column smiles --protein-column protein_sequence \
  --label-column label --split-column split --batch-size 8
scrdti train --config configs/paper/drugbank.yaml \
  --data data/pairs.npz --output runs/drugbank.pt --device cuda
scrdti predict --config configs/paper/drugbank.yaml \
  --data data/pairs.npz --checkpoint runs/drugbank.pt \
  --output runs/drugbank_test.csv --split 2 --device cuda
scrdti evaluate --predictions runs/drugbank_test.csv
```

`precompute` downloads the three Hugging Face encoder weights on first use.
Benchmark files, identifier mapping, graph/KGE artifacts, and fold generation
remain dataset-specific; see `docs/DATA.md` before reporting paper results.
