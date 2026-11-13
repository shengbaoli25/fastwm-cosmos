# Weak spatial query redundancy loss

The plug insertion diagnostic found nearly identical routing among query slots
within some reader heads. This change adds a training-only penalty on the actual
reader Q/K attention, with no new parameters or predictor layers. It is an
experimental regularizer, not evidence of improved contact localization.

For each example, reader layer, head and camera, compute spatial probabilities
`p_i = softmax(Q_i K_camera^T / sqrt(d))` over valid patches in that camera.
Average `max(0, cosine(p_i, p_j) - 0.95)` over unordered query pairs and eligible
views, heads, layers and examples. A camera needs at least two valid patches.
The two cameras are normalized separately, so changing camera preference cannot
satisfy the loss. Shared regions and overlapping attention remain allowed.

The new recipe uses weight `0.01`, linearly warmed up over 2,500 updates. The
maximum added loss after warmup is `0.0005`. Existing action/force/video losses,
online stop-gradient Force feature supervision, wrench weight, encoder LoRA,
data, batch, learning rate and 30k-stop/50k-scheduler recipe remain unchanged.
The duplicate penalty is computed in FP32 alongside the original SDPA output;
gradients reach the real query projection, key projection and observed features.
Both cameras have equal loss weight even when one has little global attention.
This is deliberate to prevent evasion through camera mass, but it may encourage
unhelpful diversity in an irrelevant view; verify task loss and localization.

Launch the opt-in matched ViT-L experiment with:

```bash
bash scripts/cosmos/launch_jepa_query_redundancy.sh
```

It selects `configs/experiments/jepa_query_redundancy.json` through the existing
official-initialization launcher. Use a new training output directory. No GPU
training run is launched by merely adding this code. The original selected
launcher keeps the original objective. The model API defaults the new weight
to zero, retaining old manifests, state keys, loss logs and checkpoint loading.
An enabled run binds the new objective into its checkpoint manifest; it must
not be presented as a strict continuation of an old objective's optimizer state.

The existing 30,000-step checkpoint used for the exhaustive query census has
**not** been trained with this regularizer. Its heatmaps are the before-change
baseline. Retraining is required before claiming any improvement.

New logs when enabled:

- `loss_query_redundancy`: unweighted mean hinge loss.
- `query_redundancy_weight`: current warmup-adjusted weight.
- `query_redundancy_cosine`: mean pair similarity, conditioned on camera.
- `query_redundancy_active_fraction`: fraction of eligible pairs above margin.

Cosine diversity alone does not break an exactly symmetric stationary point,
identify contact, guarantee useful queries, or establish robot success. The
observed queries were near-duplicates rather than exactly tied. Tests check
finite gradients and loss reduction for perturbed duplicates, disabled-path
compatibility, BF16 behavior, masks, original policy outputs, activation
checkpointing and optimizer/warmup restoration. CUDA training and a matched
retrained task-success comparison remain necessary before a benefit claim.

After training, compare the same checkpoint steps, observed-only inputs and
inference noise seeds. Report redundancy together with action error, visible
interface localization and matched interface/background occlusion. A lower
redundancy score alone is not an improvement criterion; do not optimize a loss
against labels generated from its own attention maps.
