import json
import torch
from torch import Tensor, nn

class ForceRuntimeState(nn.Module):
    """Persistent per-rank memory RNG and strict protocol bytes for ZeRO resume.

    Tensor buffers survive accelerator/DeepSpeed full-state save/load without a
    custom trainer hook or non-Tensor safetensors members. The reviewed ranks
    share seed44, batch16 and draw geometry; their states must remain identical.
    ZeRO1 saves one replicated model state, so this is not a rank-specific RNG
    persistence protocol. A different manifest is rejected before restoration.
    """
    def __init__(self, seed, manifest):
        super().__init__()
        self.register_buffer("rng_state", torch.Generator(device="cpu").manual_seed(seed).get_state())
        self.register_buffer("manifest_bytes", self._encode(manifest))
        self.register_load_state_dict_pre_hook(self._validate_restore)

    @staticmethod
    def _encode(manifest):
        data = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":")).encode()
        return torch.tensor(list(data), dtype=torch.uint8)

    def bind_manifest(self, manifest):
        self.manifest_bytes = self._encode(manifest).to(self.manifest_bytes.device)

    def _validate_restore(self, module, state_dict, prefix, *args):
        incoming = state_dict.get(prefix + "manifest_bytes")
        rng = state_dict.get(prefix + "rng_state")
        if incoming is None or not torch.equal(incoming.cpu(), self.manifest_bytes.cpu()):
            raise ValueError("Full-state Force protocol/source/initialization manifest differs")
        if not isinstance(rng, Tensor) or rng.dtype != torch.uint8 or rng.shape != self.rng_state.shape:
            raise ValueError("Full-state Force memory RNG is missing or malformed")
        # A fresh CPU generator validates the actual state, not merely its shape.
        torch.Generator(device="cpu").set_state(rng.cpu())

    def get_state(self):
        return self.rng_state.detach().cpu().clone()

    def set_state(self, state):
        torch.Generator(device="cpu").set_state(state.cpu())
        if state.shape != self.rng_state.shape:
            raise ValueError("Memory RNG shape differs")
        self.rng_state.copy_(state.to(self.rng_state.device))
        return self

    def manual_seed(self, seed):
        return self.set_state(torch.Generator(device="cpu").manual_seed(seed).get_state())

    def normal(self, shape):
        generator = torch.Generator(device="cpu").set_state(self.get_state())
        result = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
        self.set_state(generator.get_state())
        return result
