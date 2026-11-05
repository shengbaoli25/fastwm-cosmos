"""Import the actual JEPA runtime without loading weights or initializing CUDA."""
import importlib
from importlib.metadata import version
import json
import sys


def main():
    # These are timm's runtime dependencies plus the official encoder's einops.
    packages = {
        'torch': 'torch', 'torchvision': 'torchvision', 'yaml': 'PyYAML',
        'huggingface_hub': 'huggingface-hub', 'safetensors': 'safetensors',
        'einops': 'einops', 'timm': 'timm',
    }
    modules = {name: importlib.import_module(name) for name in packages}
    versions = {package: version(package) for package in packages.values()}
    if versions['timm'] != '1.0.15':
        raise RuntimeError('JEPA requires the verified timm==1.0.15 runtime')
    # Test the import which failed on the training host, not just package metadata.
    importlib.import_module('fastwam.cosmos._vendor.vjepa21.vision_transformer')
    if modules['torch'].cuda.is_initialized():
        raise RuntimeError('Dependency check must not initialize CUDA')
    print(json.dumps(dict(status='jepa_runtime_imports_verified',
                          executable=sys.executable, versions=versions,
                          timm_path=modules['timm'].__file__, cuda_initialized=False)), flush=True)


if __name__ == '__main__':
    main()
