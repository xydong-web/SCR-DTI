# SCR-DTI architecture

SCR-DTI factors evidence by **decision function**.

## 1. Pair-semantic evidence

Offline MoLFormer, MolT5, and ESM-2 pooled vectors are projected into a common
space and L2-normalized. HCA first conditions the two molecular views on the
protein representation and then forms a pair token over the active modality
tokens. The pair logit is the sum of a direct molecule-protein head and an HCA
pair-token head.

The release keeps the historical projector and HCA topology required for paper
checkpoint compatibility, but exposes it as the paper-facing `SCRDTI` API.

## 2. Independent graph-context support

When graph context is available, a dedicated linear support head maps the graph
feature vector to a logit `z_ctx`. The base decision is

`z_base = z_pair + z_ctx`.

Availability masks force `z_ctx = 0` when graph input is missing. Historical
development code also contained graph adapters and graph-expert branches; those
are not part of this clean release path.

## 3. Confidence-margin KGE refinement

The KGE score is standardized with training-partition statistics. A learned
adapter and calibration network form a candidate residual. Its influence is
scaled by

`g = mask * sigmoid(b - |z_base| / T)`

and then bounded:

`r = L * tanh(shrink * g * candidate / L)`.

The final logit is `z = z_base + r`. The gate is a deterministic confidence
proxy based on the base-logit margin, not an epistemic-uncertainty estimate.

