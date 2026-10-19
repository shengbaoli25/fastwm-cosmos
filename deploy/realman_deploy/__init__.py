"""Portable RealMan deployment runtime for the joint Cosmos checkpoint.

The deploy package deliberately lives in the training repository.  Hardware
drivers are optional adapters; importing this package never opens a socket,
camera, or robot connection.
"""

from .contracts import ActionChunk, Observation, RuntimeStatus
from .runtime import DeploymentRuntime

__all__ = ["ActionChunk", "Observation", "RuntimeStatus", "DeploymentRuntime"]
