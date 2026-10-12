# FastWAM · Cosmos28 · ForceWM28

This experiment replaces the Wan video branch with the **official Cosmos-Predict2.5-2B Base**, retains an Action Expert and a supervised autoregressive ForceWM, and aligns all three branches to **28 layers**. The implementation inherits the LIBERO two-camera FastWAM data processing and Accelerator/ZeRO trainer. Legacy Wan entry points remain in the tree for provenance; this experiment uses only `scripts/cosmos/` and `configs/task/cosmos_forcewm_4gpu_30k.yaml`.

**Status:** implementation and synthetic CPU contract tests exist. Actual official-weight loading, complete Cosmos feature export, four-GPU training, distributed checkpoint restore and evaluation must be established by runtime receipts. CPU tests are not GPU/quality evidence. See [design and operations](docs/cosmos/design.md).

| Setting | This experiment |
|---|---|
| Video | Official Cosmos-Predict2.5-2B Base, 28 blocks, width2048, 16×128 attention |
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
| Free video | 16 Euler generation steps + final clock0 feature refresh; detached/no-grad |

Only Cosmos pretrained modules are loaded. AE, Force, proprio projection and Action's two memory connections are new. **No previous Wan/Action/Force checkpoint is inherited.** This is an initialization difference as well as a video-backbone replacement; report it when comparing training curves.

```bash
git clone --recurse-submodules https://github.com/shengbaoli25/fastwm-cosmos.git
cd fastwm-cosmos
# Use the existing pinned FastWAM Torch2.7.1+cu128 environment on the GPU host.
python -m pip install -r requirements-cosmos.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/cosmos
```

The source pin and exact official asset hashes are in `src/fastwam/cosmos/assets.py`, `native.py` and `reason1_weights.json`. Never replace Base with action-conditioned Cosmos, Wan VAE features, Wan T5 features or random text features.

The unresolved `???` fields are deliberate: training requires the identity of an actual complete Cosmos feature/text cache. Follow the ordered preparation, probe, restore and foreground launch commands in [the operation guide](docs/cosmos/operations.md). Existing RGB and numeric caches are read only. VAE/text encoding happens offline; the training model has no VAE or RGB decoder.
