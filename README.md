# FastWAM · Cosmos28 · ForceWM28

This experiment replaces the Wan video branch with the **official Cosmos-Predict2.5-2B robot/action-cond**, retains an Action Expert and a supervised autoregressive ForceWM, and aligns all three branches to **28 layers**. The implementation inherits the LIBERO two-camera FastWAM data processing and Accelerator/ZeRO trainer. Legacy Wan entry points remain in the tree for provenance; this experiment uses only `scripts/cosmos/` and `configs/task/cosmos_forcewm_4gpu_30k.yaml`.

**Status:** implementation and synthetic CPU contract tests exist. Actual official-weight loading, complete Cosmos feature export, four-GPU training, distributed checkpoint restore and evaluation must be established by runtime receipts. CPU tests are not GPU/quality evidence. See [design and operations](docs/cosmos/design.md).

| Setting | This experiment |
|---|---|
| Video | Official Cosmos-Predict2.5-2B robot/action-cond, 28 blocks, width2048, 16×128 attention |
| Action Expert | Random initialization, 28 blocks, width1024, FFN4096, 16×128 attention |
| ForceWM | Random initialization, 28 blocks, width1280, FFN4608, 16 heads |
| Force objective | Shifted teacher forcing + causal AR generation, normalized six-dimensional wrench MSE; no Force diffusion |
| Images | Main left / wrist right, each224², combined224×448; nine frames at offsets0:33:4 |
| Offline video features | Independent nine-frame `[16,3,28,56]` plus independently encoded current frame `[16,1,28,56]` |
| Text | Official Cosmos-Reason1-7B, full concatenation of28 normalized hidden layers,512×100352; cached offline |
| Batch | 4 ranks ×16 per rank ×accumulation1 =64; eight-card allocation does not change this |
| Updates / save | 30000 / every5000 |
| Optimizer | AdamW1e-4, betas(.9,.95), weight_decay.01; cosine,1500 warmup |
| Precision / distribution | BF16 with native Cosmos FP32 modulation/head regions; original ZeRO1 |
| Seeds | global/AE42, Force43, generated-video memory44 |
| Losses | VideoFM + ActionFM + ForceMSE, coefficients1/1/1 |
| Free video | Current Action-conditioned, 16 Euler steps + clock0 refresh per Action evaluation; detached/no-grad |
| Control mapping | Random shared Linear32→7, grouping all 32 commands into 8 disjoint intervals with validity bits; seed45 |

Only Cosmos pretrained modules are loaded. AE, Force, proprio projection, Action's two memory connections and the dataset-to-control projection are random. Both official action embedders load from the robot checkpoint. **No previous Wan/Action/Force checkpoint is inherited.** This is an initialization difference as well as a video-backbone replacement; report it when comparing training curves.

```bash
git clone --recurse-submodules https://github.com/shengbaoli25/fastwm-cosmos.git
cd fastwm-cosmos
# Use the existing pinned FastWAM Torch2.7.1+cu128 environment on the GPU host.
python -m pip install -r requirements-cosmos.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/cosmos
```

The source pin and exact official asset hashes are in `src/fastwam/cosmos/assets.py`, `native.py` and `reason1_weights.json`. The action-conditioned checkpoint is required; Base, Wan VAE/T5 features and random text features are not substitutes.

Video supervision uses clean ground-truth actions only in its own loss path. Free video memory uses current noisy/generated Action; it is recomputed before every Action velocity and again at final A0, when Force is refreshed using the same video. At 16 Action steps, inference performs 17 video generations (289 full video passes including feature refresh), plus one observation pass. Training uses 19 video passes per batch; this added action conditioning requires a new GPU probe.

The unresolved `???` fields are deliberate: training requires the identity of an actual complete Cosmos feature/text cache. Follow the ordered preparation, probe, restore and foreground launch commands in [the operation guide](docs/cosmos/operations.md). Existing RGB and numeric caches are read only. VAE/text encoding happens offline; the training model has no VAE or RGB decoder.
