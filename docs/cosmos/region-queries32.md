# 32 independent visual queries

This optional model is based on the clean-action RealAction configuration. Select
`task=cosmos_forcewm_regions32_4gpu_30k`; the original default is unchanged.
Both world models always receive clean GT Action[0..31]. Only the Action Expert
diffuses Action. Force predicts the whole [1..32] chunk from measured history
[-31..0], clean Action, and generated Video features.

## Selection and interaction

At Cosmos layer13 (zero-based), reshape the two future latent groups into two
camera grids, each 14x14. The total future memory is 2 times x 2 cameras x 196 =
784 tokens. A single set of 32 learned query identities reads both cameras,
yielding 32 features per future time (64 total). The two cameras share each
query's probability normalization; there are no paired endpoints and no forced
16/16 camera split.

Queries depend on clean Action, time and visual content. Coarse content attention
sets a soft spatial centre, followed by a learnable bounded Gaussian locality
bias. Values pool visual features without a direct Action/query residual. This
reduces a direct control shortcut but does not prove semantic contact learning.

From layers13..27, one shared region/Force attention module exchanges region and
Force features and writes a gated spatial residual back to Video. The original
dense Video-to-Force reads stay available. One shared AE region attention module
reads the selected regions at the same layers. Shared depth weights keep the
added parameter count small; this does not eliminate repeated computation.

## Gradient boundaries

The objective is still VideoFM + ForceMSE + ActionFM with weights1:1:1. No feature
reconstruction, diversity, region annotations, or contact masks are added.

- Force/Video losses update their WMs and the region selection/fusion path.
- AE uses `shared_reader(detached_pre_fusion_video, clean_action)` rather than
  detached reader outputs or live fused WM states. AE loss updates the shared
  query reader, its small AE reading module, existing AE memory adapters,
  proprio encoder, and AE itself.
- AE cannot update Video, Force, original Video-control adapters, or the shared
  WM fusion bridge. Observation K/V and Cosmos-owned text projection are also
  disconnected from the AE graph.
- The16-step video denoising rollout remains under no_grad; Force learns through
  the final clock0 refresh only. AE never unrolls gradients through it.

Tests enforce nonzero gradients for all32 queries from Force-only and AE-only
losses, and absent gradients for all WM parameters under AE-only loss, with
checkpointing both enabled and disabled.

## Exact production parameter counts

| Component | Parameters |
| --- | ---: |
| Shared query reader (including32 query vectors) | 3,140,672 |
| Shared region/Force bridge | 4,463,874 |
| Shared AE region read | 1,575,425 |
| Total added | 9,179,971 |
| Original model including all three branches | 3,751,013,676 |
| New total | 3,760,193,647 |

The32 query vectors themselves contain16,384 parameters. Total addition is
0.244733% of the baseline. Counts use the actual production dimensions on meta
tensors, not a GPU throughput/memory benchmark.

## Training and offline inspection

Use the existing4-rank distributed launcher, batch16/rank, accumulation1,
global64, BF16/ZeRO1, lr1e-4, weight decay.01, warmup1500/cosine, max30000,
save5000. Start from official Cosmos weights with fresh AE/Force/query modules;
old RealAction or paired-query checkpoints cannot be silently resumed. Before
formal training the launcher requires two real Adam updates, full-state save,
and independent4-rank restore. The preflight checks all32 region queries too.

`scripts/cosmos/evaluate_query_regions.py` requires the exact new checkpoint
schema/manifest and supplied checkpoint SHA256. It exports all32 maps, selected
query and overview MP4s, contact sheets, numeric predictions, and a standalone
HTML query selector. Metrics in this inspection CLI use normalized training
space; complete physical-unit reports must retain the original FP64/raw-data
alignment and per-window Hold protocol.

RGB offsets are0,4,8,...,32. The first frame has no future query map; frames1..4
share future latent group1 and frames5..8 share group2. There are only two
independent future maps per window, not32 independently observed positions.
The optional reference row is decoded GT latent, not the raw recording.
Attention depicts the weights actually used to read spatial features; it is
not proof of true contact, causal importance, or persistent object identity.
First validation must inspect the maps and compare Force/Action/video quality
against the matching RealAction baseline.
