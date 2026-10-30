"""Deterministic CPU-only hardware for protocol and scheduling validation."""
import threading
import time

from .contracts import Observation


class MockHardware:
    def __init__(self, *, width=640, height=480, camera_names=("global", "wrist"),
                 force_preprocessing_id=None, force_preprocessing=None):
        self.width, self.height = width, height
        self.camera_names = tuple(camera_names)
        self.force_preprocessing_id = force_preprocessing_id
        self.force_preprocessing = force_preprocessing
        self.lock = threading.Lock()
        self.sequence = 0
        self.state = (0.0,) * 7 + (0.5,)
        self.armed = False
        self.closed = False
        self.commands = []
        self.stop_count = 0
        self.frames = {name: bytes([i * 32 + 64]) * width * height * 3
                       for i, name in enumerate(camera_names)}

    def status(self):
        return dict(connected=not self.closed, action_space="joint8", robot_ready=True,
                    cameras_ready=True, force_ready=True, gripper_ready=True, armed=self.armed,
                    force_preprocessing=self.force_preprocessing or (
                        {"id": self.force_preprocessing_id} if self.force_preprocessing_id else None))

    def observe_at(self, anchor):
        with self.lock:
            if self.closed:
                raise RuntimeError("mock hardware is closed")
            sequence = self.sequence
            self.sequence += 1
            state = self.state
        meta = {name: dict(name=name, width=self.width, height=self.height,
                          color_order="RGB", received=anchor, frame_number=sequence)
                for name in self.camera_names}
        force_meta = dict(source_time=anchor, grid_time=anchor, source_age_s=0.0, epoch=0)
        if self.force_preprocessing_id:
            force_meta["preprocessing_id"] = self.force_preprocessing_id
        return Observation(anchor, sequence, state, (0.0,) * 6, self.frames, meta,
                           force_meta, "joint8")

    def arm(self):
        if self.closed:
            raise RuntimeError("mock hardware is closed")
        self.armed = True

    def command(self, action):
        with self.lock:
            if not self.armed or self.closed:
                raise RuntimeError("mock hardware is not armed")
            self.state = tuple(action)
            self.commands.append((time.monotonic(), self.state))

    def slow_stop(self):
        self.armed = False
        self.stop_count += 1

    def close(self):
        self.armed = False
        self.closed = True
