"""Export a no-reference runtime from the resolved training config and checkpoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'deploy')]

from fastwam.cosmos.noaction import SCHEMA
from realman_deploy.preprocessing import binding, file_sha256


def runtime_config(config, manifest, *, checkpoint, normalization, tokenizer,
                   hardware_config, execution_mode='synchronous', sampling_profile=None,
                   device='cuda', optimization='auto'):
    """Check recorded identities before constructing a deployable configuration."""
    model = dict(config['model'])
    if model.pop('_target_') != 'fastwam.cosmos.assets.create_noaction_joint_cosmos':
        raise ValueError('Resolved no-reference training config required')
    for config_key, manifest_key in (
            ('expected_data_identity', 'dataset_identity'),
            ('expected_text_identity', 'text_identity'),
            ('expected_normalization_sha256', 'normalization_sha256')):
        value = model.get(config_key)
        if not value or value != manifest.get(manifest_key):
            raise ValueError('Training config and checkpoint differ: ' + config_key)
    if file_sha256(normalization) != manifest['normalization_sha256']:
        raise ValueError('Normalization differs from the training-bound file')
    from realman_deploy.normalization import DatasetNormalizer
    normalizer = DatasetNormalizer.from_json(normalization)
    hardware = json.loads(Path(hardware_config).read_text())
    if (manifest.get('action_space') != 'joint8' or model.get('action_space') != 'joint8'
            or normalizer.action_space != 'joint8' or hardware.get('action_space') != 'joint8'):
        raise ValueError('Joint8 checkpoint, normalization and hardware configuration required')
    for key, value in manifest['extra_config'].items():
        if model.get(key) != value:
            raise ValueError('Training model settings differ: ' + key)
    text_root = Path(model['text_cache']).expanduser().resolve()
    text = json.loads((text_root / 'manifest.json').read_text())
    prompts = [entry['prompt'] for entry in text['texts']]
    if len(prompts) != 1:
        raise ValueError('RealMan runtime export requires a single recorded task prompt')
    for key in ('upstream', 'checkpoint', 'text_cache'):
        model[key] = str(Path(model[key]).expanduser().resolve())
    if sampling_profile is None:
        if execution_mode not in {'synchronous', 'async_relaxed', 'timestamped'}:
            raise ValueError('Unsupported execution mode')
        backend = 'joint_static_cache'
    else:
        from realman_deploy.sampling import PROFILES, profile_steps
        if sampling_profile not in PROFILES:
            raise ValueError('Sampling profile must be 161616 or 4416')
        profile_steps(sampling_profile)
        if execution_mode not in PROFILES[sampling_profile]['execution_modes']:
            raise ValueError(f'{sampling_profile} does not support execution mode {execution_mode}')
        backend = 'joint_multirate'
    if not (device == 'cuda' or (device.startswith('cuda:') and device[5:].isdigit())):
        raise ValueError('Device must be cuda or a logical cuda:N ordinal')
    if optimization not in {'auto', 'eager', 'compile', 'cuda_graph'}:
        raise ValueError('Unsupported optimization mode')
    result = dict(device=device, execution_mode=execution_mode,
        normalization_file=str(Path(normalization).resolve()),
        hardware_factory='realman_deploy.hardware:make_hardware',
        hardware_config=str(Path(hardware_config).resolve()), tokenizer=str(Path(tokenizer).resolve()),
        camera_order=['global', 'wrist'], camera_width=640, camera_height=480,
        joint_steps=manifest['num_inference_steps'], inference_backend=backend,
        action_seed=42, video_seed=44, force_seed=43, model_kwargs=model,
        task_prompt=prompts[0], checkpoint_directory=str(Path(checkpoint).resolve().parent))
    if sampling_profile is not None:
        result['sampling_profile'] = sampling_profile
        result['inference_optimization'] = {'mode': optimization}
    return result


def main():
    import torch
    from omegaconf import OmegaConf
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-config', 'checkpoint', 'normalization', 'tokenizer', 'hardware-config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sampling-profile', choices=['161616', '4416', 'legacy'], default='legacy',
                   help='Runtime sampling profile; legacy preserves the old single-step-count sampler')
    p.add_argument('--execution-mode', choices=['synchronous', 'async_relaxed', 'timestamped'], default='synchronous')
    p.add_argument('--device', default='cuda', help='CUDA logical device, e.g. cuda:1')
    p.add_argument('--optimization', choices=['auto', 'eager', 'compile', 'cuda_graph'], default='auto')
    for name in ('upstream', 'base-checkpoint', 'text-cache'):
        p.add_argument('--' + name, type=Path, help='Optional relocated asset path on the deployment host')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output path; do not overwrite a live runtime configuration')
    if args.sampling_profile == 'legacy' and args.execution_mode == 'async_relaxed':
        p.error('async_relaxed is available only for profile 4416')
    if args.sampling_profile == '161616' and args.execution_mode == 'async_relaxed':
        p.error('async_relaxed is available only for profile 4416')
    for path in (args.tokenizer, args.hardware_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('schema') != SCHEMA:
        raise ValueError('This exporter requires a trained no-reference diffusion checkpoint')
    config = OmegaConf.to_container(OmegaConf.load(args.training_config), resolve=True, throw_on_missing=True)
    for arg, key in ((args.upstream, 'upstream'), (args.base_checkpoint, 'checkpoint'), (args.text_cache, 'text_cache')):
        if arg is not None:
            config['model'][key] = str(arg.resolve())
    profile = None if args.sampling_profile == 'legacy' else args.sampling_profile
    result = runtime_config(config, payload['manifest'], checkpoint=args.checkpoint,
        normalization=args.normalization, tokenizer=args.tokenizer,
        hardware_config=args.hardware_config, execution_mode=args.execution_mode,
        sampling_profile=profile, device=args.device, optimization=args.optimization)
    metadata = json.loads(args.normalization.read_text())
    if metadata.get('force_preprocessing') is not None:
        receipt = binding(args.checkpoint, args.normalization)
        sidecar = Path(str(args.checkpoint) + '.force.json')
        if sidecar.exists() and json.loads(sidecar.read_text()) != receipt:
            raise ValueError('Existing checkpoint preprocessing binding differs')
        if not sidecar.exists():
            sidecar.write_text(json.dumps(receipt, indent=2) + '\n')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
