"""Single-GPU, observed-only JEPA contact-attention diagnostic (no training).

The sample inventory and raw contact sheets are written before model construction.
Contact annotations are external to the attention maps. Missing annotations never
become fabricated contact scores. All78 was used for training: this is a mechanism
diagnostic on training-distribution trajectories, not a held-out success metric.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
VIEWS = ('cam_main', 'cam_wrist')
OBSERVED_KEYS = ('first_frame_latents', 'history', 'history_is_pad',
                 'history_channel_valid', 'cosmos_text_index', 'jepa_observation')
PHASES = (('early', .2), ('middle', .5), ('late', .8))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def progress(output, phase, **extra):
    event = dict(status='running', phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'progress.json', event)
    print(json.dumps(event, allow_nan=False), flush=True)


def print_report_summary(report):
    print(json.dumps(dict(status='evaluation_report',
        completed_baselines=report['completed_baselines'], planned_samples=report['planned_samples'],
        contact_annotation_status=report['contact_annotation_status'],
        annotation_temporal_scope=report.get('annotation_temporal_scope'),
        contact_ci_by_view_and_layer=report['contact_cluster_ci_by_view_and_layer'],
        matched_region_ci=report['matched_region_cluster_ci'],
        intervention_ci=report['intervention_cluster_ci'],
        scope='training-distribution descriptive diagnostic; not held-out task success'), allow_nan=False), flush=True)


def select_anchors(rows, episodes=12):
    """Select at fixed trajectory fractions, without consulting RGB or attention."""
    if episodes < 1:
        raise ValueError('Positive episode count required')
    by_episode = {}
    for index, (ep, frame, lo, hi) in enumerate(rows):
        by_episode.setdefault(int(ep), []).append((index, int(frame), int(lo), int(hi)))
    available = sorted(by_episode)
    if not available:
        raise ValueError('No valid source anchors')
    chosen = np.linspace(0, len(available) - 1, min(episodes, len(available)), dtype=int)
    result = []
    for pos in chosen:
        ep = available[int(pos)]
        anchors = by_episode[ep]
        for phase, fraction in PHASES:
            index, frame, lo, hi = anchors[round((len(anchors) - 1) * fraction)]
            result.append(dict(sample_index=index, episode_index=ep, frame_index=frame,
                               segment_start=lo, segment_stop=hi, phase=phase,
                               observed_frame_indices=[max(lo, frame - 4), frame]))
    if len({row['sample_index'] for row in result}) != len(result):
        raise ValueError('Selected episodes need at least three distinct anchors')
    return result


def observed_input(sample):
    """Batch ONE anchor, explicitly dropping every future-target tensor."""
    result = {key: sample[key].unsqueeze(0) for key in OBSERVED_KEYS}
    proprio = sample['proprio']
    if proprio.ndim != 2 or proprio.shape[1] != 9:
        raise ValueError('Expected Joint8 proprio [horizon,9]')
    result['proprio'] = proprio[0].unsqueeze(0).clone()
    return result


def intervention_positions(rows, count=12):
    """One anchor per evenly sampled episode, early/late alternating prospectively."""
    if count < 0:
        raise ValueError('Nonnegative intervention sample count required')
    episodes = sorted({row['episode_index'] for row in rows})
    selected = np.linspace(0, len(episodes) - 1, min(count, len(episodes)), dtype=int)
    positions = []
    for ordinal, selected_index in enumerate(selected):
        episode = episodes[int(selected_index)]
        phase = 'early' if ordinal % 2 == 0 else 'late'
        matching = [index for index, row in enumerate(rows) if row['episode_index'] == episode and row['phase'] == phase]
        if len(matching) != 1:
            raise ValueError('Exactly one selected early/late anchor per episode required')
        positions.append(matching[0])
    return positions


def roi_grid(entry, side=14):
    """Fractional union area of annotation rectangles, one view's patch grid."""
    result = np.zeros((side, side), dtype=np.float32)
    if entry.get('status') != 'visible':
        return result
    boxes = entry.get('boxes', [])
    if not boxes:
        raise ValueError('Visible contact requires at least one rectangle')
    # Exact rectangular union integration through all annotation/patch boundaries.
    parsed = []
    for box in boxes:
        if len(box) != 4 or not np.isfinite(box).all():
            raise ValueError('Finite normalized xyxy contact rectangles required')
        x0, y0, x1, y1 = map(float, box)
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('Contact rectangle is outside normalized RGB')
        parsed.append((x0, y0, x1, y1))
    xs = sorted(set(np.linspace(0, 1, side + 1).tolist() + [v for b in parsed for v in (b[0], b[2])]))
    ys = sorted(set(np.linspace(0, 1, side + 1).tolist() + [v for b in parsed for v in (b[1], b[3])]))
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            x, y = (x0 + x1) / 2, (y0 + y1) / 2
            if any(a <= x <= c and b <= y <= d for a, b, c, d in parsed):
                result[min(side - 1, int(y * side)), min(side - 1, int(x * side))] += (x1 - x0) * (y1 - y0) * side * side
    return np.clip(result, 0, 1)


def matching_random_mask(mask, rng, forbidden=None):
    """Exactly match patch count PER CAMERA; optionally exclude contact patches."""
    result = np.zeros_like(mask, dtype=bool)
    forbidden = np.zeros_like(mask, dtype=bool) if forbidden is None else forbidden
    for view in range(len(mask)):
        count = int(mask[view].sum())
        available = np.flatnonzero(~forbidden[view].reshape(-1))
        if count > len(available):
            raise ValueError('Insufficient non-contact area for matched background control')
        result[view].reshape(-1)[rng.choice(available, count, replace=False)] = True
    return result


def matching_translated_mask(mask, rng, forbidden=None):
    """Translate the entire ROI mask per view, preserving area and topology exactly.

    Choose uniformly among feasible disjoint translations on the patch lattice.
    Fail rather than silently substitute a differently shaped control.
    """
    if mask.ndim != 3 or mask.dtype != bool:
        raise ValueError('Boolean [views,rows,columns] mask required')
    forbidden = mask if forbidden is None else forbidden
    if forbidden.shape != mask.shape:
        raise ValueError('Forbidden area geometry differs')
    result = np.zeros_like(mask)
    for view in range(len(mask)):
        ys, xs = np.nonzero(mask[view])
        if not len(ys):
            continue
        ys, xs = ys - ys.min(), xs - xs.min()
        candidates = []
        for row in range(mask.shape[1] - int(ys.max())):
            for col in range(mask.shape[2] - int(xs.max())):
                if not forbidden[view, ys + row, xs + col].any():
                    candidates.append((row, col))
        if not candidates:
            raise ValueError(f'No disjoint same-shape background translation for camera {view}')
        row, col = candidates[int(rng.integers(len(candidates)))]
        result[view, ys + row, xs + col] = True
    return result


def occlude_observation(observation, mask):
    """Zero normalized RGB inside selected patches in BOTH observed tubelet frames."""
    import torch
    side = mask.shape[-1]
    if mask.shape != (2, side, side) or observation.shape != (1, 2, 3, 2, 224, 224) or 224 % side:
        raise ValueError('Two square patch grids and one observed RGB example required')
    pixels = np.repeat(np.repeat(mask, 224 // side, axis=1), 224 // side, axis=2)
    pixels = torch.as_tensor(pixels, device=observation.device)[None, :, None, None]
    return observation.masked_fill(pixels, 0)


@contextmanager
def zero_residual(model):
    import torch
    gates = model.dit['jepa_adapter'].gates
    old = gates.detach().clone()
    try:
        with torch.no_grad():
            gates.zero_()
        yield
    finally:
        with torch.no_grad():
            gates.copy_(old)


def target_errors(prediction, sample):
    import torch
    action_valid = ~sample['action_is_pad'].bool()
    force_valid = sample['wrench_valid'].bool() & ~sample['force_is_pad'].bool()[:, None]
    a = prediction['action'][0].detach().float().cpu()
    f = prediction['force'][0].detach().float().cpu()
    def mse(x, target, mask):
        mask = mask.expand_as(x)
        return float((x - target.float()).square()[mask].mean()) if bool(mask.any()) else None
    if not torch.isfinite(a).all() or not torch.isfinite(f).all():
        raise ValueError('Nonfinite inference result')
    return dict(action_mse_normalized=mse(a, sample['action'], action_valid[:, None]),
                force_mse_normalized=mse(f, sample['wrench_future'], force_valid))


def rgb_array(observation, view, frame):
    return ((observation[view, :, frame].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def save_raw_sheet(path, sample, row):
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (448, 498), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 3), f"sample {row['sample_index']} ep {row['episode_index']} frame {row['frame_index']}", fill='black')
    for time_index in range(2):
        for view in range(2):
            x, y = view * 224, 30 + time_index * 234
            sheet.paste(Image.fromarray(rgb_array(sample['jepa_observation'], view, time_index)), (x, y))
            draw.text((x + 4, y - 12), f'{VIEWS[view]} observed {row["observed_frame_indices"][time_index]}', fill='black')
    sheet.save(path)


def heat_sheet(path, sample, row, attention):
    from PIL import Image, ImageDraw
    calls = sorted(set((0, len(attention) // 2, len(attention) - 1)))
    depth = attention.shape[1]
    result = Image.new('RGB', (448 * depth, 248 * len(calls) + 28), 'white')
    draw = ImageDraw.Draw(result)
    draw.text((4, 4), 'Mean over query/head; tubelet [-4,0]; RED scale fixed: 0 to 10/392 (clipped)', fill='black')
    for row_index, call in enumerate(calls):
        for layer in range(depth):
            maps = attention[call, layer].mean(axis=(0, 1)).reshape(2, 14, 14)
            xbase, ybase = layer * 448, 28 + row_index * 248
            draw.text((xbase + 4, ybase), f'ep {row["episode_index"]} f {row["frame_index"]} call {call} layer {layer}', fill='black')
            for view in range(2):
                heat = np.asarray(Image.fromarray(maps[view], mode='F').resize((224, 224), Image.Resampling.BILINEAR))
                strength = np.clip(heat / (10 / 392), 0, 1)[..., None] * .7
                raw = rgb_array(sample['jepa_observation'], view, 1)
                colored = (raw * (1 - strength) + np.array([255, 0, 0]) * strength).astype(np.uint8)
                result.paste(Image.fromarray(colored), (xbase + view * 224, ybase + 20))
    result.save(path)


def prepare(args, dataset):
    output = args.output
    rows = select_anchors(dataset.rows, args.episodes)
    inventory = dict(schema='jepa-contact-eval-samples-v1', selection=f'{args.episodes} evenly spaced episodes; temporal fractions .2/.5/.8; no attention selection',
                     inference_target_keys_excluded=True, train_distribution_only=True,
                     base_cache_identity=dataset.identity_sha256,
                     observation_cache_identity=dataset.observation_identity_sha256, samples=rows)
    manifest = output / 'samples.json'
    if manifest.exists() and json.loads(manifest.read_text()) != inventory:
        raise ValueError('Existing sample manifest differs; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'raw').mkdir(exist_ok=True)
    atomic_json(manifest, inventory)
    roi = dict(schema='jepa-contact-roi-v1', annotation_stage='UNANNOTATED', samples={})
    for row in rows:
        sample = dataset[row['sample_index']]
        save_raw_sheet(output / 'raw' / f'{row["sample_index"]:06d}.png', sample, row)
        roi['samples'][str(row['sample_index'])] = {view: dict(status='unannotated', boxes=[]) for view in VIEWS}
    if not (output / 'roi-template.json').exists():
        atomic_json(output / 'roi-template.json', roi)
    return rows


def summarize(output, roi_path=None):
    """Score frozen saved attention after annotations, without running the model."""
    from fastwam.cosmos.jepa_contact_eval import cluster_bootstrap
    rows = json.loads((output / 'samples.json').read_text())['samples']
    annotations = json.loads(Path(roi_path).read_text()) if roi_path else None
    records = []
    if annotations is not None:
        if annotations.get('schema') != 'jepa-contact-roi-v1':
            raise ValueError('Unknown contact annotation schema')
        for row in rows:
            path = output / 'attention' / f'{row["sample_index"]:06d}.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as saved:
                weights = saved['attention']
            for view, name in enumerate(VIEWS):
                entry = annotations.get('samples', {}).get(str(row['sample_index']), {}).get(name, {})
                roi = roi_grid(entry)
                if not roi.any():
                    continue
                # Within-view metrics prevent camera choice from masquerading as localization.
                maps = weights.reshape(*weights.shape[:-1], 2, 14, 14)[..., view, :, :].mean(axis=(2, 3))
                for layer in range(maps.shape[1]):
                    current = maps[:, layer]
                    view_mass = current.sum(axis=(-2, -1))
                    conditional = current / np.maximum(view_mass[:, None, None], 1e-30)
                    mass = (conditional * roi).sum(axis=(-2, -1))
                    peaks = conditional.reshape(len(conditional), -1).argmax(-1)
                    records.append(dict(**row, view=name, layer_index=layer,
                        contact_mass=float(mass.mean()), contact_area_fraction=float(roi.mean()),
                        enrichment=float((mass / roi.mean()).mean()),
                        pointing_hit=float((roi.reshape(-1)[peaks] > 0).mean()),
                        view_mass=float(view_mass.mean()), annotation_status=entry['status']))
    metrics_path = output / 'metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    contrasts = []
    for metric in metrics:
        if metric['condition'] == 'baseline':
            continue
        base = next(x for x in metrics if x['sample_index'] == metric['sample_index'] and x['condition'] == 'baseline')
        contrasts.append(dict(episode_index=metric['episode_index'], condition=metric['condition'],
            sample_index=metric['sample_index'],
            delta_action_mse_normalized=metric['action_mse_normalized'] - base['action_mse_normalized']))
    matched_contrasts = []
    for targeted, control in (('top_attention', 'matched_random'), ('contact_roi', 'matched_background')):
        for metric in [m for m in metrics if m['condition'] == targeted]:
            comparator = next((m for m in metrics if m['sample_index'] == metric['sample_index'] and m['condition'] == control), None)
            if comparator is not None:
                matched_contrasts.append(dict(episode_index=metric['episode_index'], sample_index=metric['sample_index'],
                    contrast=f'{targeted}_minus_{control}',
                    excess_action_mse_normalized=metric['action_mse_normalized'] - comparator['action_mse_normalized']))
    result = dict(schema='jepa-contact-eval-report-v1', generated_at=datetime.now(timezone.utc).isoformat(),
        completed_baselines=sum(x['condition'] == 'baseline' for x in metrics), planned_samples=len(rows),
        contact_annotation_status='available' if records else 'pending_or_no_visible_contact',
        contact_records=records, paired_interventions=contrasts, matched_region_contrasts=matched_contrasts,
        limitations=['All78 training-distribution trajectories; no held-out generalization claim.',
                     'Cross-attention is reader routing, not a contact detector or causal explanation by itself.',
                     'One JEPA tubelet fuses two frames; no independent past/current temporal attention map.',
                     'Occlusions modify only JEPA RGB; original Cosmos observed latents stay fixed.',
                     'Zero-residual is an ablation of this trained model, not a separately trained Cosmos baseline.',
                     'Top-attention versus scattered-random patches matches area but may differ in occlusion topology.',
                     'Contact versus background uses same-shape translated masks, with no pair when a disjoint translation is impossible.',
                     'Only visible annotated interfaces contribute contact metrics; a small number of episodes is exploratory.',
                     'Normalized open-loop action error is not physical robot success.'],
        annotation_stage=annotations.get('annotation_stage') if annotations else None,
        annotation_temporal_scope=annotations.get('temporal_scope') if annotations else None,
        inference_observed_keys=list(OBSERVED_KEYS) + ['proprio_current_only'])
    result['contact_cluster_ci'] = cluster_bootstrap(records, 'enrichment', n_bootstrap=2000, seed=113) if records else None
    result['contact_cluster_ci_by_view_and_layer'] = {f'{view}/layer{layer}': cluster_bootstrap(
        [r for r in records if r['view'] == view and r['layer_index'] == layer],
        'enrichment', n_bootstrap=2000, seed=113) for view, layer in sorted({(r['view'], r['layer_index']) for r in records})}
    result['intervention_cluster_ci'] = {condition: cluster_bootstrap([r for r in contrasts if r['condition'] == condition],
        'delta_action_mse_normalized', n_bootstrap=2000, seed=113) for condition in sorted({r['condition'] for r in contrasts})}
    result['matched_region_cluster_ci'] = {contrast: cluster_bootstrap([r for r in matched_contrasts if r['contrast'] == contrast],
        'excess_action_mse_normalized', n_bootstrap=2000, seed=113) for contrast in sorted({r['contrast'] for r in matched_contrasts})}
    atomic_json(output / 'report.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, help='Completed training run directory containing resolved.yaml/training/')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['prepare', 'evaluate', 'summarize'], default='evaluate')
    p.add_argument('--episodes', type=int, default=12)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--seed', type=int, default=113)
    p.add_argument('--intervention-samples', type=int, default=12)
    p.add_argument('--roi', type=Path)
    args = p.parse_args(argv)
    try:
        run(args)
    except BaseException as error:
        failure = dict(status='failed', error_type=type(error).__name__, error=str(error),
            updated_at=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc())
        print(json.dumps(failure), flush=True)
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output / 'failure.json', failure)
            if args.mode != 'summarize':
                atomic_json(args.output / 'progress.json', failure)
        except OSError:
            pass
        raise


def run(args):
    if args.mode == 'summarize':
        print_report_summary(summarize(args.output, args.roi))
        return
    if args.run_root is None:
        raise ValueError('--run-root is required for prepare/evaluate')
    if args.steps < 1 or args.intervention_samples < 0:
        raise ValueError('Positive steps and nonnegative intervention sample count required')
    if args.mode == 'evaluate' and (args.output / 'metrics.jsonl').exists():
        raise FileExistsError('Evaluation output already contains metrics; use fresh output to avoid mixing runs')
    progress(args.output, 'import_dependencies', mode=args.mode)
    import torch
    from omegaconf import OmegaConf
    from fastwam.cosmos.jepa_training_data import JepaForceTrainingDataset
    from fastwam.cosmos.jepa_force_latent import create_jepa_force_latent_cosmos
    from fastwam.cosmos.jepa_contact_eval import AttentionCapture, token_layout
    from PIL import Image
    progress(args.output, 'verify_local_inputs')
    cfg_path = args.run_root / 'resolved.yaml'
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Training resolved configuration is missing: {cfg_path}')
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True, throw_on_missing=True)
    checkpoint = args.checkpoint or args.run_root / 'training/checkpoints/weights/step_030000.pt'
    if args.mode == 'evaluate':
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
            raise FileNotFoundError(f'Completed step30000 checkpoint is missing or empty: {checkpoint}')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Exactly one visible CUDA device required')
        for key in ('checkpoint', 'text_cache', 'upstream'):
            if not Path(cfg['model'][key]).exists():
                raise FileNotFoundError(f'Pinned local model asset {key} is missing: {cfg["model"][key]}')
        if not Path(cfg['model']['observation_encoder']['checkpoint']).is_file():
            raise FileNotFoundError('Pinned official JEPA encoder checkpoint is missing')
        print(json.dumps(dict(status='checkpoint_file_present', checkpoint=str(checkpoint),
            bytes=checkpoint.stat().st_size, strict_schema_and_step_verification='pending_model_load',
            gpu=torch.cuda.get_device_name())), flush=True)
    if args.roi is not None and not args.roi.is_file():
        raise FileNotFoundError(f'ROI annotation file is missing: {args.roi}')
    data_cfg = dict(cfg['data']['train'])
    if data_cfg.pop('_target_') != 'fastwam.cosmos.jepa_training_data.JepaForceTrainingDataset':
        raise ValueError('This evaluator requires the exact plug JEPA dataset')
    data_cfg.pop('_recursive_', None)
    progress(args.output, 'verify_dataset_and_rgb_cache', note='Hash verification can take several minutes')
    dataset = JepaForceTrainingDataset(**data_cfg)
    progress(args.output, 'prepare_blinded_rgb_sheets', valid_anchors=len(dataset))
    rows = prepare(args, dataset)
    chosen = set(intervention_positions(rows, args.intervention_samples))
    atomic_json(args.output / 'intervention-plan.json', dict(
        schema='jepa-contact-intervention-plan-v1', selection='one anchor per evenly sampled episode; early/late alternating before inference',
        requested_count=args.intervention_samples, count=len(chosen),
        samples=[row for index, row in enumerate(rows) if index in chosen]))
    if args.mode == 'prepare':
        print(json.dumps(dict(status='prepared_raw_rgb', samples=len(rows))), flush=True)
        return
    model_cfg = dict(cfg['model'])
    if model_cfg.pop('_target_') != 'fastwam.cosmos.jepa_force_latent.create_jepa_force_latent_cosmos':
        raise ValueError('Exact trained JEPA factory required')
    model_cfg.pop('_recursive_', None)
    model_cfg['device'] = 'cuda'
    progress(args.output, 'construct_model_from_pinned_local_assets', samples=len(rows))
    model = create_jepa_force_latent_cosmos(**model_cfg)
    progress(args.output, 'strict_restore_trained_checkpoint', checkpoint=str(checkpoint))
    step = model.load_checkpoint(checkpoint)
    if step != 30000:
        raise ValueError('Expected completed plug insertion step 30000')
    model.eval()
    model.requires_grad_(False)
    model.gradient_checkpointing = False
    progress(args.output, 'hash_checkpoint_and_record_provenance')
    manifest = dict(schema='jepa-contact-eval-run-v1', checkpoint=str(checkpoint), checkpoint_step=step,
        checkpoint_sha256=sha256(checkpoint), config_path=str(cfg_path), config_sha256=sha256(cfg_path),
        samples_sha256=sha256(args.output / 'samples.json'), gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__, inference_steps=args.steps, seed=args.seed,
        contact_roi_path=str(args.roi) if args.roi else None,
        contact_roi_sha256=sha256(args.roi) if args.roi else None,
        variant=model_cfg['observation_encoder']['variant'], token_layout=token_layout(),
        residual_gates=model.dit['jepa_adapter'].gates.detach().float().cpu().tolist(),
        model_manifest=model.manifest, train_distribution_only=True)
    atomic_json(args.output / 'run.json', manifest)
    for name in ('attention', 'heatmaps', 'predictions', 'masks'):
        (args.output / name).mkdir(exist_ok=True)
    annotations = json.loads(args.roi.read_text()) if args.roi else None
    if annotations and annotations.get('schema') != 'jepa-contact-roi-v1':
        raise ValueError('Unknown contact annotation schema')
    metrics_file = args.output / 'metrics.jsonl'
    isolation_tolerance = dict(force=1e-5, video_latents=1e-5)
    for position, row in enumerate(rows):
        started = time.monotonic()
        sample = dataset[row['sample_index']]
        observed = observed_input(sample)
        infer_kw = dict(num_inference_steps=args.steps, seed=args.seed, force_seed=args.seed + 1, video_seed=args.seed + 2, use_cache=True)
        progress(args.output, 'baseline_inference', completed=position, total=len(rows), sample_index=row['sample_index'])
        with torch.inference_mode(), AttentionCapture(model.dit['jepa_reader']) as capture:
            baseline = model.infer_joint(observed, **infer_kw)
        if position == 0:
            progress(args.output, 'check_fixed_seed_repeatability', sample_index=row['sample_index'])
            with torch.inference_mode():
                repeated = model.infer_joint(observed, **infer_kw)
            repeatability = {key: float((repeated[key].float() - baseline[key].float()).abs().max())
                             for key in ('action', 'force', 'video_latents')}
            for key in isolation_tolerance:
                isolation_tolerance[key] = max(1e-5, 2 * repeatability[key])
            atomic_json(args.output / 'repeatability.json', dict(seed=args.seed,
                sample_index=row['sample_index'], maximum_absolute_difference=repeatability,
                isolation_tolerance=isolation_tolerance,
                interpretation='Numerical diagnostic only; underlying graph should isolate Force/Video exactly'))
            print(json.dumps(dict(status='fixed_seed_repeatability', **repeatability)), flush=True)
            del repeated
        records = capture.records
        depth = len(model.dit['jepa_reader'].readers)
        if len(records) != args.steps * depth:
            raise ValueError('Attention capture did not cover every denoise call and reader layer')
        if any(not bool(record['valid'].all()) for record in records):
            raise ValueError('This spatial report requires all patches valid; masked-patch scoring must be explicit')
        attention = torch.stack([r['attention'][0] for r in records]).reshape(args.steps, depth, *records[0]['attention'].shape[1:]).numpy()
        np.savez_compressed(args.output / 'attention' / f'{row["sample_index"]:06d}.npz', attention=attention,
            action_clock=torch.stack([r['action_clock'][0] for r in records]).reshape(args.steps, depth).numpy(),
            call_index=np.asarray([r['call_index'] for r in records]).reshape(args.steps, depth),
            layer_index=np.asarray([r['layer_index'] for r in records]).reshape(args.steps, depth),
            valid=torch.stack([r['valid'][0] for r in records]).reshape(args.steps, depth, -1).numpy(),
            observed_frame_indices=np.asarray(row['observed_frame_indices']))
        heat_sheet(args.output / 'heatmaps' / f'{row["sample_index"]:06d}.png', sample, row, attention)
        def record(condition, result):
            metric = dict(**row, condition=condition, **target_errors(result, sample))
            metric['action_change_rms_normalized'] = float((result['action'].float() - baseline['action'].float()).square().mean().sqrt())
            metric['force_change_max_normalized'] = float((result['force'].float() - baseline['force'].float()).abs().max())
            metric['video_latent_change_max'] = float((result['video_latents'].float() - baseline['video_latents'].float()).abs().max())
            if (metric['force_change_max_normalized'] > isolation_tolerance['force']
                    or metric['video_latent_change_max'] > isolation_tolerance['video_latents']):
                raise ValueError('JEPA-only intervention changed isolated Force/Video beyond measured numerical repeatability')
            metric['elapsed_seconds'] = time.monotonic() - started
            with metrics_file.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metric, allow_nan=False) + '\n')
            np.savez_compressed(args.output / 'predictions' / f'{row["sample_index"]:06d}-{condition}.npz',
                action=result['action'].detach().float().cpu().numpy(), force=result['force'].detach().float().cpu().numpy())
            return metric
        record('baseline', baseline)
        if position in chosen:
            rng = np.random.default_rng(args.seed + row['sample_index'])
            scores = attention.mean(axis=(0, 1, 2, 3)).reshape(2, 14, 14)
            top = np.zeros_like(scores, dtype=bool)
            for view in range(2):
                top[view].reshape(-1)[np.argsort(scores[view].reshape(-1), kind='stable')[-20:]] = True
            masks = dict(top_attention=top, matched_random=matching_random_mask(top, rng))
            mask_metadata = dict(schema='jepa-contact-perturbation-mask-v1', sample_index=row['sample_index'],
                rng_seed=args.seed + row['sample_index'], patch_size_pixels=16,
                observed_frame_indices=row['observed_frame_indices'], applies_to='both_observed_frames_jepa_only',
                normalized_rgb_fill=0, contact_control='same_shape_disjoint_translation_per_camera',
                top_attention_control='equal_area_scattered_random; topology_not_matched',
                roi_sha256=manifest['contact_roi_sha256'], skipped_contact_reason=None,
                color_scale=[1.15, .9, .8])
            if annotations:
                entries = annotations.get('samples', {}).get(str(row['sample_index']), {})
                fractions = np.stack([roi_grid(entries.get(view, {})) for view in VIEWS])
                contact = fractions > 0
                if contact.any():
                    try:
                        background = matching_translated_mask(contact, rng, forbidden=contact)
                    except ValueError as error:
                        mask_metadata['skipped_contact_reason'] = str(error)
                    else:
                        masks['contact_roi'] = contact
                        masks['matched_background'] = background
                else:
                    mask_metadata['skipped_contact_reason'] = 'No visible annotated interface in either camera'
            else:
                mask_metadata['skipped_contact_reason'] = 'No ROI annotation supplied'
            mask_metadata['patch_counts_by_camera'] = {name: mask.sum(axis=(1, 2)).tolist() for name, mask in masks.items()}
            np.savez_compressed(args.output / 'masks' / f'{row["sample_index"]:06d}.npz', **masks)
            atomic_json(args.output / 'masks' / f'{row["sample_index"]:06d}.json', mask_metadata)
            for name, mask in masks.items():
                progress(args.output, 'intervention_inference', completed=position, total=len(rows),
                    sample_index=row['sample_index'], condition=name)
                changed = dict(observed, jepa_observation=occlude_observation(observed['jepa_observation'], mask))
                with torch.inference_mode():
                    record(name, model.infer_joint(changed, **infer_kw))
            with torch.inference_mode(), zero_residual(model):
                progress(args.output, 'intervention_inference', completed=position, total=len(rows),
                    sample_index=row['sample_index'], condition='zero_jepa_residual')
                record('zero_jepa_residual', model.infer_joint(observed, **infer_kw))
            color = observed['jepa_observation'].clone()
            color = ((color + 1) * .5 * color.new_tensor([1.15, .9, .8])[None, None, :, None, None, None]).clamp(0, 1) * 2 - 1
            with torch.inference_mode():
                progress(args.output, 'intervention_inference', completed=position, total=len(rows),
                    sample_index=row['sample_index'], condition='rgb_color_shift')
                record('rgb_color_shift', model.infer_joint(dict(observed, jepa_observation=color), **infer_kw))
        atomic_json(args.output / 'progress.json', dict(status='running', completed=position + 1, total=len(rows),
            last_sample=row['sample_index'], updated_at=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(status='sample_complete', completed=position + 1, total=len(rows), sample=row['sample_index'], elapsed_seconds=time.monotonic() - started)), flush=True)
    print_report_summary(summarize(args.output, args.roi))
    atomic_json(args.output / 'progress.json', dict(status='complete', completed=len(rows), total=len(rows),
        updated_at=datetime.now(timezone.utc).isoformat()))
    print('JEPA_CONTACT_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
