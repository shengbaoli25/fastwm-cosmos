"""Loopback-only web dashboard and checkpoint download API."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import copy
import os
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

from .checkpoints import CheckpointCatalog, CheckpointStore
from .model_adapter import CosmosJointPolicy
from .runtime import DeploymentRuntime
from .sampling import settings, updated_config
from .deployment_settings import deployment_settings, update_deployment_settings


STATIC = Path(__file__).with_name('static')


class DeploymentWeb:
    def __init__(self, runtime: DeploymentRuntime, *, catalog_path: str | Path,
                 config_path: str | Path, hardware_factory: str | None = None,
                 host: str = "127.0.0.1", port: int = 8765):
        self.runtime = runtime
        self.catalog = CheckpointCatalog(catalog_path)
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text())
        checkpoint_root = self.config.get("checkpoint_directory")
        if checkpoint_root:
            checkpoint_root = Path(checkpoint_root).expanduser()
            if not checkpoint_root.is_absolute():
                checkpoint_root = self.config_path.parent / checkpoint_root
        self.store = CheckpointStore(self.catalog, checkpoint_root)
        self.hardware_factory = hardware_factory
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.download_lock = threading.Lock()
        self.connect_lock = threading.Lock()
        self.config_lock = threading.Lock()
        self.serving = threading.Event()

    def settings(self):
        return {**settings(self.config), **deployment_settings(self.config),
                "supported": self.runtime.normalizer.action_space == "joint8"}

    def select_policy(self, ident: str) -> str:
        """Select an exact model locally or validate it on the remote service."""
        with self.config_lock:
            selected = copy.deepcopy(self.config)
            remote = selected.get("inference_device", "local") == "qizhi_h200"
            path = self.store.path_for(ident) if ident else None
            spec = self.catalog.specs.get(ident)
            if remote:
                from .remote_transport import RemoteJointPolicy
                options = selected.setdefault("remote_inference", {})
                configured_sha = options.get("checkpoint_sha256")
                if spec and spec.sha256:
                    if configured_sha and configured_sha != spec.sha256:
                        raise ValueError("所选模型与远端期望权重 SHA256 不一致")
                    options["checkpoint_sha256"] = spec.sha256
                if not options.get("checkpoint_sha256") and not (path and path.is_file()):
                    raise ValueError("请提供所选权重的 SHA256，或选择已下载的本地权重用于身份校验")
                checkpoint = str(path) if path and path.is_file() else None
                label = path.name if path else "远端模型"
                def factory(_):
                    return RemoteJointPolicy(checkpoint, selected, config_path=self.config_path)
            else:
                cuda = self.runtime._cuda_status()
                if not cuda.get("available"):
                    raise RuntimeError(cuda.get("error") or "CUDA 正在检查，请等待检查完成后再加载模型")
                if path is None or not path.is_file():
                    raise FileNotFoundError("checkpoint is not downloaded yet")
                label = path.name
                def factory(_):
                    from .checkpoints import sha256_file
                    if spec and spec.sha256 and sha256_file(path) != spec.sha256:
                        raise ValueError("local checkpoint SHA256 differs from catalog")
                    return CosmosJointPolicy(path, selected, config_path=self.config_path)
            self.runtime.load_policy_async(factory, label)
            return label

    def _hardware_config_path(self):
        path = Path(self.config['hardware_config']).expanduser()
        return path if path.is_absolute() else self.config_path.parent / path

    def _new_debug_gripper(self):
        from .gripper import Ctag2F120Worker
        config = json.loads(self._hardware_config_path().read_text())
        if not isinstance(config, dict) or not config.get('robot_ip') or type(config.get('robot_port')) is not int:
            raise ValueError('夹爪连接需要有效 robot_ip / robot_port')
        return Ctag2F120Worker(config, diagnostic=True)

    def _save_config(self, config):
        """Atomic replace in the config directory, preserving all unrelated keys."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.config_path.parent,
                                             prefix=".runtime-", suffix=".json", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(config, stream, ensure_ascii=False, indent=2)
                stream.write(chr(10)); stream.flush(); os.fsync(stream.fileno())
            temporary.chmod(self.config_path.stat().st_mode & 0o777)
            os.replace(temporary, self.config_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _handler(self):
        web = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "FastWAMDeploy/1"

            def log_message(self, *_args):
                pass

            def _json(self, value, code=200):
                data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(data)

            def _body(self):
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 1024 * 1024:
                    raise ValueError("request body too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                return body

            def _local_origin(self):
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise PermissionError("cross-origin control request rejected")

            def do_GET(self):
                assets = {"/": ("index.html", "text/html"),
                          "/static/app.js": ("app.js", "text/javascript"),
                          "/static/style.css": ("style.css", "text/css")}
                if self.path in assets:
                    name, content_type = assets[self.path]
                    body = (STATIC / name).read_bytes(); self.send_response(200)
                    self.send_header("Content-Type", content_type + "; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                    self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(body)
                elif self.path == "/api/status":
                    self._json(web.runtime.status())
                elif self.path == "/api/settings":
                    with web.config_lock:
                        self._json(web.settings())
                elif self.path == "/api/gripper":
                    self._json(web.runtime.gripper_debug_status())
                elif self.path == "/api/checkpoints":
                    self._json({"directory": str(web.store.root), "checkpoints": web.store.list()})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                try:
                    self._local_origin(); body = self._body()
                    if self.path == "/api/gripper":
                        unknown = set(body) - {'action', 'opening', 'speed', 'torque'}
                        if unknown:
                            raise ValueError('未知夹爪参数: ' + ', '.join(sorted(unknown)))
                        action = body.get('action')
                        if not isinstance(action, str):
                            raise ValueError('缺少夹爪 action')
                        self._json(web.runtime.debug_gripper(action, factory=web._new_debug_gripper,
                            opening=body.get('opening'), speed=body.get('speed'), torque=body.get('torque')))
                    elif self.path == "/api/download":
                        with web.download_lock:
                            path = web.store.download(str(body.get("id", "")))
                        self._json({"path": str(path)})
                    elif self.path == "/api/select":
                        label = web.select_policy(str(body.get("id", "")))
                        self._json({"state": "loading", "checkpoint": label}, 202)
                    elif self.path == "/api/settings":
                        with web.config_lock:
                            if not web.settings()["supported"]:
                                raise ValueError("三专家采样设置仅适用于 Joint8 模型")
                            updated = updated_config(web.config, body.get("sampling_profile"), body.get("execution_mode"))
                            updated = update_deployment_settings(updated, body)
                            web.runtime.configure_sampling(updated["sampling_profile"], updated["execution_mode"],
                                persist=lambda: web._save_config(updated),
                                inference_device=updated["inference_device"],
                                max_result_age_s=updated["max_result_age_s"],
                                action_smoothing_alpha=updated["action_smoothing_alpha"])
                            web.config = updated
                            self._json({**web.settings(), "reload_required": True})
                    elif self.path == "/api/connect":
                        factory = web.hardware_factory
                        if not factory:
                            raise RuntimeError("未配置硬件插件。设置 FASTWAM_HARDWARE_FACTORY=module:factory 后重启服务。")
                        from .adapters import load_factory
                        hardware_config = web._hardware_config_path()
                        if not web.connect_lock.acquire(blocking=False):
                            raise RuntimeError("硬件连接正在进行")
                        try:
                            status = web.runtime.connect_hardware(
                                lambda: load_factory(factory, config_path=str(hardware_config)))
                            self._json({"state": "connected", "hardware": status})
                        finally:
                            web.connect_lock.release()
                    elif self.path == "/api/start":
                        web.runtime.start(); self._json({"state": "running"})
                    elif self.path == "/api/pause":
                        web.runtime.pause(); self._json({"state": "paused"})
                    elif self.path == "/api/estop":
                        web.runtime.emergency_stop(); self._json({"state": "estop"})
                    elif self.path == "/api/estop/reset":
                        if set(body) != {"confirmed"} or body["confirmed"] is not True:
                            raise ValueError("恢复急停需要 confirmed: true，且不可包含其他字段")
                        self._json(web.runtime.reset_emergency_stop(confirmed=True))
                    else:
                        self._json({"error": "not found"}, 404)
                except FileNotFoundError as exc:
                    self._json({"error": str(exc)}, 404)
                except PermissionError as exc:
                    self._json({"error": str(exc)}, 403)
                except (ValueError, KeyError, RuntimeError, OSError) as exc:
                    self._json({"error": str(exc)}, 400)

        return Handler

    def serve_forever(self):
        self.serving.set()
        try:
            self.server.serve_forever(poll_interval=0.25)
        finally:
            self.serving.clear()
            self.server.server_close()

    def close(self):
        if self.serving.is_set():
            self.server.shutdown()
        self.server.server_close()
        self.runtime.close()
