"""Small real CUDA allocation/matmul check; no model download or robot access."""
import json
import platform

import torch
import torchvision
from torch.nn import functional as F

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Linux environment")
device = torch.device("cuda:0")
left = torch.ones((64, 64), device=device, dtype=torch.bfloat16)
result = left @ left
torch.cuda.synchronize(device)
if not bool(torch.all(result == 64)):
    raise SystemExit("CUDA BF16 matmul produced an unexpected result")
query = torch.randn((1, 4, 32, 64), device=device, dtype=torch.bfloat16)
attention = F.scaled_dot_product_attention(query, query, query, dropout_p=0.)
torch.cuda.synchronize(device)
if attention.shape != query.shape or not bool(torch.isfinite(attention).all()):
    raise SystemExit("CUDA BF16 SDPA output is invalid")
print(json.dumps(dict(platform=platform.platform(), torch=torch.__version__,
                     torchvision=torchvision.__version__, cuda_runtime=torch.version.cuda,
                     device=torch.cuda.get_device_name(device), capability=torch.cuda.get_device_capability(device),
                     total_vram_bytes=torch.cuda.get_device_properties(device).total_memory,
                     bf16_matmul_passed=True, bf16_sdpa_passed=True), indent=2))
