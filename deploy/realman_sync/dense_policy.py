"""Online DINOv3 / V-JEPA Joint8 adapter; training modules stay unchanged.

Both policies consume measured 30 Hz history and the two RGB observations at
source offsets [-4, 0]. The two images are never synthesized from one frame or
from two consecutive (and potentially seconds apart) HTTP requests.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path


_TEACHERS = {
    'dinov3_vith16plus': (1280, 'adjacent_frame_mean_v1'),
    'vjepa21_vitg': (1408, 'official_video_tubelet2'),
    'vjepa21': (768, 'official_video_tubelet2'),
}
_MODEL_TARGET = 'fastwam.cosmos.dense_features.create_dense_feature_joint'
_ARCHITECTURE = ('feature_dim', 'hidden_dim', 'depth', 'heads', 'view_grid',
                 'action_hidden', 'action_ffn', 'force_hidden', 'force_ffn',
                 'joint_heads', 'joint_head_dim', 'context_dim')
_REQUIRED_MODEL = _ARCHITECTURE + ('manifest', 'lambda_future', 'force_shift',
    'action_seed', 'force_seed', 'proprio_seed', 'gradient_checkpointing',
    'num_inference_steps')


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{label} must be a positive integer')
    return value


class DensePolicy:
    """Resident teacher + trained policy, returning raw absolute Joint8 actions.

    Required deployment paths are relative to ``config_path``: training_config
    (the run's saved training/config.yaml), normalization_file (the run's
    training-only normalization.json), teacher_source and teacher_weights.
    Also supply force_preprocessing_file (the dataset meta/force.json) or its
    model_input contract as force_preprocessing. The existing teacher loader
    verifies the official weight SHA256 and pinned source revision.
    """

    def __init__(self, config: dict, config_path: str | Path, checkpoint: str | Path):
        self.config = deepcopy(config)
        self.config_dir = Path(config_path).expanduser().resolve().parent
        self.checkpoint = self._path(checkpoint)
        self.camera_names = list(config.get('camera_order', ['global', 'wrist']))
        if (len(self.camera_names) != 2 or len(set(self.camera_names)) != 2
                or any(not isinstance(x, str) or not x for x in self.camera_names)):
            raise ValueError('camera_order must name main camera then wrist camera exactly once')
        self.width = _positive_int(config.get('camera_width', 640), 'camera_width')
        self.height = _positive_int(config.get('camera_height', 480), 'camera_height')
        self.action_horizon = _positive_int(config.get('action_horizon', 32), 'action_horizon')
        if self.action_horizon > 32:
            raise ValueError('The trained action horizon is at most 32')
        self.action_space = 'joint8'
        self.control_hz = 30
        self.history_length = 32
        self.video_context_frames = 2
        self.video_context_stride = 4
        self.device = str(config.get('device', 'cuda'))
        self.model = self.teacher = None
        self.inference_index = 0
        try:
            self._load()
        except BaseException:
            self.close()
            raise
        self.metadata = dict(camera_names=self.camera_names, width=self.width,
            height=self.height, history_length=self.history_length,
            video_context_frames=self.video_context_frames,
            video_context_stride=self.video_context_stride,
            control_hz=self.control_hz, action_horizon=self.action_horizon,
            action_space=self.action_space, force_preprocessing=deepcopy(self.force_preprocessing),
            teacher=self.teacher_name, observation_offsets=[-4, 0],
            action_units=['rad'] * 7 + ['openness_0_1'])

    def _path(self, value) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.config_dir / path).resolve()

    def _required_path(self, key, *, directory=False):
        if not self.config.get(key):
            raise ValueError(f'Dense deployment requires {key}')
        path = self._path(self.config[key])
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(f'{key} does not exist: {path}')
        return path

    def _model_configuration(self, trained, manifest):
        """Require saved run kwargs, including force_shift absent from manifest."""
        if not isinstance(trained, dict) or not isinstance(trained.get('model'), dict):
            raise ValueError('training_config must be the resolved saved run configuration')
        model = deepcopy(trained['model'])
        if model.pop('_target_', None) != _MODEL_TARGET:
            raise ValueError('training_config must describe the dense Joint8 model')
        missing = [name for name in _REQUIRED_MODEL if name not in model]
        if missing:
            raise ValueError('Saved run model configuration is incomplete: ' + ', '.join(missing))
        if not isinstance(model['manifest'], dict) or not model['manifest']:
            raise ValueError('Use saved training/config.yaml, not the unresolved experiment template')
        for name, value in model['manifest'].items():
            if manifest.get(name) != value:
                raise ValueError(f'Training config and checkpoint manifest differ: {name}')
        for name in _ARCHITECTURE:
            if model[name] != manifest.get('architecture', {}).get(name):
                raise ValueError(f'Training config and checkpoint architecture differ: {name}')
        for name in ('lambda_future', 'num_inference_steps'):
            if model[name] != manifest.get(name):
                raise ValueError(f'Training config and checkpoint differ: {name}')
        teacher = manifest.get('teacher', {})
        if teacher.get('name') not in _TEACHERS or trained.get('teacher') != teacher.get('name'):
            raise ValueError('Checkpoint and training_config teacher differ or are unsupported')
        width, pooling = _TEACHERS[teacher['name']]
        if (teacher.get('feature_dim') != width or model['feature_dim'] != width
                or manifest.get('action_space') != 'joint8'
                or manifest.get('observation_offsets') != [-4, 0]
                or manifest.get('future_offsets') != [4, 8, 12, 16, 20, 24, 28, 32]
                or manifest.get('stride') != 4
                or manifest.get('temporal_pooling') != pooling
                or manifest.get('temporal_slots') != [[-4, 0], [4, 8], [12, 16], [20, 24], [28, 32]]
                or manifest.get('normalization') != 'training episodes only'
                or not manifest.get('dataset_source_sha256')
                or not manifest.get('cache_identity_sha256')):
            raise ValueError('Checkpoint temporal/teacher/normalization contract is incompatible')
        if not math.isfinite(float(model['force_shift'])) or float(model['force_shift']) <= 0:
            raise ValueError('Saved force_shift must be positive and finite')
        model['gradient_checkpointing'] = False
        return model

    def _load_normalizer(self, manifest):
        from realman_deploy.normalization import DatasetNormalizer
        from realman_force.contract import validate_contract_identity

        payload = json.loads(self._required_path('normalization_file').read_text())
        if (payload.get('cache_identity') != manifest['cache_identity_sha256']
                or payload.get('normalization') != 'training episodes only'
                or payload.get('train_episodes') != manifest.get('splits', {}).get('train')):
            raise ValueError('normalization_file must contain the matching run training-only statistics')
        force = self.config.get('force_preprocessing')
        if self.config.get('force_preprocessing_file'):
            saved = json.loads(self._required_path('force_preprocessing_file').read_text())
            saved = saved.get('model_input', saved)
            if force is not None and force != saved:
                raise ValueError('Inline and file force preprocessing contracts differ')
            force = saved
        if not isinstance(force, dict):
            raise ValueError('Dense deployment requires the training dataset force preprocessing contract')
        validate_contract_identity(force)
        self.force_preprocessing = deepcopy(force)
        self.normalizer = DatasetNormalizer(payload.get('statistics', {}), force, 'joint8')

    def _load(self):
        import torch
        from omegaconf import OmegaConf
        from fastwam.cosmos.dense_encoder import load_dense_teacher
        from fastwam.cosmos.dense_features import DenseFeatureJointModel, SCHEMA

        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'Trained policy checkpoint does not exist: {self.checkpoint}')
        config_path = self._required_path('training_config')
        teacher_source = self._required_path('teacher_source', directory=True)
        teacher_weights = self._required_path('teacher_weights')
        trained = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if (not isinstance(payload, dict) or payload.get('schema') != SCHEMA
                or not isinstance(payload.get('manifest'), dict)
                or not isinstance(payload.get('model'), dict)):
            raise ValueError('Expected a dense Joint8 weights checkpoint, not a distributed optimizer state')
        manifest = payload['manifest']
        kwargs = self._model_configuration(trained, manifest)
        self._load_normalizer(manifest)
        self.teacher_name = manifest['teacher']['name']
        self.num_inference_steps = _positive_int(
            self.config.get('num_inference_steps', manifest['num_inference_steps']), 'num_inference_steps')
        device = torch.device(self.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; set device explicitly for CPU testing')
        dtype_name = self.config.get('model_dtype', 'bfloat16' if device.type == 'cuda' else 'float32')
        if dtype_name not in ('float32', 'bfloat16'):
            raise ValueError('model_dtype must be float32 or bfloat16')
        self.dtype = getattr(torch, dtype_name)
        # Keep construction on CPU: the unchanged scheduler computes scalar
        # training-weight constants and cannot be constructed on the meta device.
        # assign=True releases initialization tensors in favor of mmap weights
        # before transferring the loaded model to its inference device.
        model = DenseFeatureJointModel(**kwargs)
        if model.manifest != manifest:
            raise ValueError('Complete reconstructed model/checkpoint manifest differs')
        model.load_state_dict(payload['model'], strict=True, assign=True)
        self.model = model.to(device=device, dtype=self.dtype).eval()
        del payload
        self.teacher = load_dense_teacher(self.teacher_name, teacher_source, teacher_weights,
            manifest['teacher']['weight_sha256'], device=self.device)
        actual_teacher = self.teacher.manifest()
        for name, value in actual_teacher.items():
            if manifest['teacher'].get(name) != value:
                raise ValueError(f'Loaded teacher differs from training checkpoint: {name}')

    @staticmethod
    def _stamp(item):
        sequence, stamp = item.get('sequence'), item.get('stamp')
        if (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0
                or isinstance(stamp, bool) or not isinstance(stamp, (float, int))
                or not math.isfinite(stamp)):
            raise ValueError('Observation sequence and client monotonic stamp must be finite')
        return sequence, float(stamp)

    def _validate_force(self, item):
        if item.get('force_meta', {}).get('preprocessing_id') != self.force_preprocessing['id']:
            raise ValueError('Observation force preprocessing differs from the training contract')

    def _history_sample(self, observation, history):
        import torch

        if not isinstance(history, list) or not 1 <= len(history) <= 32:
            raise ValueError('Measured history must contain 1..32 samples ending at the observation')
        current = self._stamp(observation)
        if self._stamp(history[-1]) != current:
            raise ValueError('Measured history must end at the current observation')
        for field in ('state', 'force', 'force_meta'):
            if history[-1].get(field) != observation.get(field):
                raise ValueError(f'Current observation/history {field} differs')
        rows = []
        previous = None
        for item in history:
            sequence, stamp = self._stamp(item)
            if previous is not None and (sequence != previous[0] + 1
                    or abs(stamp - previous[1] - 1 / self.control_hz) > 0.015):
                raise ValueError('Measured history must be consecutive samples on the 30 Hz client clock')
            self._validate_force(item)
            state = self.normalizer.encode('state', item['state'])
            if not 0 <= float(item['state'][7]) <= 1:
                raise ValueError('Measured gripper openness must be in [0, 1]')
            rows.append(state + self.normalizer.encode('force', item['force']))
            previous = (sequence, stamp)
        pad = 32 - len(rows)
        is_pad = [True] * pad + [False] * len(rows)
        rows = [[0.] * 14 for _ in range(pad)] + rows
        sample = {
            'history': torch.tensor([rows], device=self.device, dtype=self.dtype),
            'history_is_pad': torch.tensor([is_pad], device=self.device, dtype=torch.bool),
            'history_channel_valid': torch.tensor([[[not p] * 14 for p in is_pad]],
                                                 device=self.device, dtype=torch.bool),
            'proprio': torch.tensor([self.normalizer.encode('state', observation['state']) + [0.]],
                                   device=self.device, dtype=self.dtype),
        }
        return sample

    def _encode_observed_video(self, observation, video_context):
        import torch
        from torch.nn import functional as F

        if not isinstance(video_context, list) or len(video_context) != 2:
            raise ValueError('Two measured RGB observations at source offsets [-4, 0] are required')
        older, latest = video_context
        old_sequence, old_stamp = self._stamp(older)
        sequence, stamp = self._stamp(observation)
        if self._stamp(latest) != (sequence, stamp):
            raise ValueError('Video context must end at the current observation')
        if (old_sequence != sequence - 4
                or abs(stamp - old_stamp - 4 / self.control_hz) > 0.02):
            raise ValueError('Video context must use source offsets [-4, 0] on the 30 Hz client clock')
        if latest.get('frames') != observation.get('frames'):
            raise ValueError('Video context and current observation images differ')
        paired = []
        for item in video_context:
            views = []
            for name in self.camera_names:
                frame = item.get('frames', {}).get(name)
                if (not isinstance(frame, dict) or frame.get('width') != self.width
                        or frame.get('height') != self.height
                        or not isinstance(frame.get('data'), bytes)
                        or len(frame['data']) != self.width * self.height * 3):
                    raise ValueError(f'Camera {name} must contain RGB24 {self.width}x{self.height}')
                rgb = torch.frombuffer(bytearray(frame['data']), dtype=torch.uint8)
                rgb = rgb.reshape(self.height, self.width, 3).permute(2, 0, 1).float().div_(255.)
                # Matches RealManSource.decode_episode Resize([224,224]) before
                # the teacher's own bicubic/ImageNet preprocessing.
                views.append(F.interpolate(rgb.unsqueeze(0), size=(224, 224),
                    mode='bilinear', align_corners=False, antialias=True)[0])
            paired.append(torch.cat(views, dim=-1).sub_(0.5).div_(0.5))
        pixels = torch.stack(paired, dim=1).unsqueeze(0).to(self.device)
        features = self.teacher.encode_video(pixels, context_only=True)
        expected_width = _TEACHERS[self.teacher_name][0]
        if (not isinstance(features, torch.Tensor)
                or features.shape != (1, 1, 392, expected_width)
                or not bool(torch.isfinite(features).all())):
            raise ValueError('Teacher must return finite observed features [1,1,392,D]')
        return features.to(device=self.device, dtype=self.dtype)

    def predict(self, observation: dict, history: list[dict],
                video_context: list[dict]) -> list[list[float]]:
        import torch

        if self.model is None or self.teacher is None:
            raise RuntimeError('Dense policy is closed or not loaded')
        if observation.get('action_space') != self.action_space:
            raise ValueError('Dense policy requires raw joint8 observations')
        with torch.inference_mode():
            sample = self._history_sample(observation, history)
            sample['visual_features'] = self._encode_observed_video(observation, video_context)
            result = self.model.infer_joint(sample, num_inference_steps=self.num_inference_steps,
                seed=int(self.config.get('action_seed', 42)) + self.inference_index,
                force_seed=int(self.config.get('force_seed', 43)))
            action = result.get('action') if isinstance(result, dict) else None
            if (not isinstance(action, torch.Tensor) or action.shape != (1, 32, 8)
                    or not bool(torch.isfinite(action).all())):
                raise ValueError('Dense policy must return finite normalized action[1,32,8]')
            raw = self.normalizer.decode('action', action[0, :self.action_horizon].float().cpu().tolist())
            # The training target is physical openness; diffusion can overshoot
            # this bounded actuator channel, just as in the existing adapter.
            for row in raw:
                row[7] = min(1., max(0., row[7]))
            self.inference_index += 1
            return raw

    def close(self):
        self.model = None
        self.teacher = None
