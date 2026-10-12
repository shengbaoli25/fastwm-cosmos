"""Window-aligned frozen VAE features; importing this package loads no RGB reader."""
from .latent import (CacheIntegrityError, LatentCacheDataset, cache_identity_sha256,
                     make_cache_identity, validate_cache, write_window_cache)

__all__ = ["CacheIntegrityError", "LatentCacheDataset", "cache_identity_sha256",
           "make_cache_identity", "validate_cache", "write_window_cache"]
