This encoder-only subset is copied from facebookresearch/vjepa2 at
`204698b45b3712590f06245fbfba32d3be539812`. `UPSTREAM_FILES.json` records
the original file paths and SHA256 hashes. The four imports in
`vision_transformer.py` use relative imports to avoid colliding with the
Cosmos upstream `src` and `app` packages. All model computation is unchanged.
The upstream MIT license is included as `LICENSE`.

Official checkpoint URLs (ema_encoder entries, weights are not vendored):

- https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
- https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt

The experiment runs observed two-frame clips at 224 pixels with the upstream
interpolated RoPE support. This is an explicit evaluation-resolution change
from the 384-pixel pretrained models; both experiment variants share it.
