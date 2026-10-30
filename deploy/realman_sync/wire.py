"""Versioned stdlib-only wire contract, shared verbatim with the robot client.

All timestamps belong to the client clock. Never subtract a server timestamp
from an observation timestamp. RGB is lossless; no pickle or executable payloads.
"""
from __future__ import annotations

import base64
import binascii
import math

PROTOCOL = "realman-sync-v1"
MAX_BODY_BYTES = 64 * 1024 * 1024


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def vector(value, width, name):
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{name} must contain {width} values")
    return [finite(v, name) for v in value]


def validate_actions(actions, action_space="joint8", max_horizon=32):
    if action_space != "joint8":
        raise ValueError("This protocol requires absolute joint8 actions")
    if not isinstance(actions, (list, tuple)) or not 1 <= len(actions) <= max_horizon:
        raise ValueError("Invalid action horizon")
    result = [vector(a, 8, "action") for a in actions]
    if any(not 0 <= a[-1] <= 1 for a in result):
        raise ValueError("Action gripper must be in [0, 1]")
    return result


def _sample(sample, metadata):
    if not isinstance(sample, dict):
        raise ValueError("Observation/history sample must be an object")
    sequence = integer(sample.get("sequence"), "sequence")
    stamp = finite(sample.get("stamp"), "stamp")
    if stamp <= 0:
        raise ValueError("stamp must be positive client monotonic time")
    state = vector(sample.get("state"), 8, "state")
    if not 0 <= state[-1] <= 1:
        raise ValueError("Measured gripper must be in [0, 1]")
    force = vector(sample.get("force"), 6, "force")
    force_meta = sample.get("force_meta", {})
    if not isinstance(force_meta, dict):
        raise ValueError("force_meta must be an object")
    expected = metadata.get("force_preprocessing")
    if expected is not None:
        if force_meta.get("preprocessing_id") != expected["id"]:
            raise ValueError("Force preprocessing differs from trained model")
        integer(force_meta.get("epoch"), "force epoch")
        source = finite(force_meta.get("source_time"), "force source_time")
        if not -1e-8 <= stamp - source <= 0.030001:
            raise ValueError("Force source is stale or from the future")
    return dict(sequence=sequence, stamp=stamp, state=state, force=force, force_meta=force_meta)


def encode_observation(observation):
    """Encode a raw dictionary (frames contain bytes and width/height)."""
    result = dict(observation)
    result["frames"] = {
        name: dict(width=frame["width"], height=frame["height"],
                   encoding="rgb8-base64", data=base64.b64encode(frame["data"]).decode("ascii"))
        for name, frame in observation["frames"].items()
    }
    return result


def decode_observation(observation, metadata):
    result = _sample(observation, metadata)
    if observation.get("action_space") != metadata["action_space"]:
        raise ValueError("Observation action_space differs from server")
    frames = observation.get("frames")
    if not isinstance(frames, dict) or set(frames) != set(metadata["camera_names"]):
        raise ValueError("Camera names differ from server")
    decoded = {}
    for name, frame in frames.items():
        if not isinstance(frame, dict) or frame.get("encoding") != "rgb8-base64":
            raise ValueError("Frames require rgb8-base64 encoding")
        width = integer(frame.get("width"), "width", 1, 4096)
        height = integer(frame.get("height"), "height", 1, 4096)
        if (width, height) != (metadata["width"], metadata["height"]):
            raise ValueError("Raw camera dimensions differ from server")
        encoded = frame.get("data")
        expected = width * height * 3
        if not isinstance(encoded, str) or len(encoded) != 4 * ((expected + 2) // 3):
            raise ValueError("RGB frame length differs from dimensions")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid RGB base64") from exc
        if len(raw) != expected:
            raise ValueError("Invalid RGB byte count")
        decoded[name] = dict(width=width, height=height, data=raw)
    frame_meta = observation.get("frame_meta", {})
    if not isinstance(frame_meta, dict):
        raise ValueError("frame_meta must be an object")
    received = []
    for name, meta in frame_meta.items():
        if name not in frames or not isinstance(meta, dict):
            raise ValueError("Unexpected camera metadata")
        for key, expected_value in (("name", name), ("color_order", "RGB"),
                                    ("width", metadata["width"]), ("height", metadata["height"])):
            if key in meta and meta[key] != expected_value:
                raise ValueError(f"Camera metadata {key} mismatch")
        if "received" in meta:
            stamp = finite(meta["received"], "camera received")
            anchor = result["force_meta"].get("source_time", result["stamp"])
            if not -1e-8 <= anchor - stamp <= .120001 or abs(stamp - anchor) > .050001:
                raise ValueError("Camera and robot timestamps are misaligned")
            received.append(stamp)
    if received and max(received) - min(received) > .050001:
        raise ValueError("Camera pair timestamps are misaligned")
    result.update(action_space=metadata["action_space"], frames=decoded, frame_meta=frame_meta)
    return result


def _same_sample(first, second):
    return all(first.get(k) == second.get(k) for k in ("sequence", "stamp", "state", "force", "force_meta"))


def _identifier(value, name):
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or not value.isascii() or not value.isprintable():
        raise ValueError(f"{name} must be 1..128 printable ASCII characters")
    return value


def decode_request(payload, metadata):
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported protocol")
    session_id = _identifier(payload.get("session_id"), "session_id")
    request_id = _identifier(payload.get("request_id"), "request_id")
    if payload.get("force_preprocessing") != metadata.get("force_preprocessing"):
        raise ValueError("Client and model force preprocessing identities differ")
    obs = decode_observation(payload.get("observation"), metadata)
    raw_history = payload.get("history")
    if not isinstance(raw_history, list) or not 1 <= len(raw_history) <= metadata["history_length"]:
        raise ValueError("history must contain 1..history_length samples")
    history = [_sample(s, metadata) for s in raw_history]
    if not _same_sample(history[-1], obs):
        raise ValueError("History must end at the exact current observation")
    period = 1.0 / metadata["control_hz"]
    for previous, current in zip(history, history[1:]):
        if current["sequence"] != previous["sequence"] + 1:
            raise ValueError("History must use consecutive sensor samples")
        if abs(current["stamp"] - previous["stamp"] - period) > 0.010:
            raise ValueError("History cadence must match control_hz")
        if previous["force_meta"].get("epoch") != current["force_meta"].get("epoch"):
            raise ValueError("History cannot cross force filter epochs")
    raw_context = payload.get("video_context", [payload["observation"]])
    count = metadata.get("video_context_frames", 1)
    stride = metadata.get("video_context_stride", 1)
    if not isinstance(raw_context, list) or len(raw_context) != count:
        raise ValueError("Incorrect video context frame count")
    context = [decode_observation(s, metadata) for s in raw_context]
    if not _same_sample(context[-1], obs) or context[-1]["frames"] != obs["frames"]:
        raise ValueError("Video context must end at the exact current observation")
    for i, item in enumerate(context):
        offset = (count - 1 - i) * stride
        if item["sequence"] != obs["sequence"] - offset:
            raise ValueError("Incorrect video context frame stride")
        if abs(obs["stamp"] - item["stamp"] - offset * period) > 0.010:
            raise ValueError("Video context timestamps differ from training cadence")
        if item["force_meta"].get("epoch") != obs["force_meta"].get("epoch"):
            raise ValueError("Video context cannot cross force filter epochs")
        matching = next((s for s in history if s["sequence"] == item["sequence"]), None)
        if matching is not None and not _same_sample(matching, item):
            raise ValueError("Video context and history disagree")
    return dict(session_id=session_id, request_id=request_id, observation=obs,
                history=history, video_context=context)


def validate_response(payload, request, metadata):
    if not isinstance(payload, dict):
        raise ValueError("Response must be an object")
    for key, expected in (("protocol", PROTOCOL), ("session_id", request["session_id"]),
                          ("request_id", request["request_id"]),
                          ("source_sequence", request["observation"]["sequence"]),
                          ("action_space", metadata["action_space"]),
                          ("control_hz", metadata["control_hz"])):
        if payload.get(key) != expected:
            raise ValueError(f"Response {key} mismatch")
    if type(payload.get("source_sequence")) is not int:
        raise ValueError("Response source_sequence must be an integer")
    return validate_actions(payload.get("actions"), metadata["action_space"], metadata["action_horizon"])
