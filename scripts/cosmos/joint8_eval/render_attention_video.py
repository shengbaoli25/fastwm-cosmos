"""Render sparse, time-aligned Joint8 attention comparisons on recorded RGB."""
import argparse
import json
from pathlib import Path

import numpy as np


BRANCHES = ('video', 'force', 'action')
BRANCH_LABELS = {'video': 'Video query -> observed video',
                 'force': 'Force query -> observed video',
                 'action': 'AE query -> observed video'}
VIEWS = ('global', 'wrist')
DISPLAY_GAMMA = 0.5


def load_runs(runs):
    protocols = [json.loads((run / 'protocol.json').read_text()) for run in runs]
    fields = ('source_sha256', 'source_identity', 'dataset_identity', 'video_episode',
              'windows', 'seeds', 'probe_indices')
    for protocol in protocols[1:]:
        if any(protocol[k] != protocols[0][k] for k in fields):
            raise ValueError('Evaluation protocols differ')
    if any(p['status'] != 'complete' for p in protocols):
        raise ValueError('Only complete evaluation runs can be compared')
    steps = [p['step'] for p in protocols]
    if len(steps) != 2 or steps[0] >= steps[1]:
        raise ValueError('Exactly two checkpoints are required, ordered baseline then later checkpoint')
    return protocols


def colorize(rgb, heat, vmax, boxes):
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    base = Image.fromarray(rgb.astype(np.uint8), mode='RGB').convert('RGBA')
    scaled = np.clip(heat / max(vmax, 1e-12), 0, 1)
    display = scaled ** DISPLAY_GAMMA
    cmap = matplotlib.colormaps['inferno']
    colors = (cmap(display)[..., :3] * 255).astype(np.uint8)
    alpha = (display * 170).astype(np.uint8)
    overlay = Image.fromarray(np.dstack((colors, alpha)), mode='RGBA')
    overlay = overlay.resize(base.size, Image.Resampling.NEAREST)
    result = Image.alpha_composite(base, overlay).convert('RGB')
    draw = ImageDraw.Draw(result)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline='#00e5b0', width=2)
    return np.asarray(result)


def panel(rgb, title, height=30):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (rgb.shape[1], rgb.shape[0] + height), '#15191d')
    image.paste(Image.fromarray(rgb), (0, height))
    draw = ImageDraw.Draw(image)
    draw.text((5, 4), title, fill='white', font=ImageFont.load_default())
    return image


def compose_frame(gt, maps, boxes, episode, frame, episode_time, layer, scales, steps, view):
    from PIL import Image, ImageDraw, ImageFont
    rows = []
    camera_offset = 0 if view == 'global' else 14
    gt_view = gt[:, camera_offset * 16:(camera_offset + 14) * 16]
    for branch in BRANCHES:
        scale = scales[(layer, branch)]
        gt_overlay = colorize(gt_view, np.zeros((14, 14), dtype=np.float32), scale, boxes)
        row = [panel(gt_overlay, f'Ep{episode:03d} f{frame} t={episode_time:.2f}s GT')]
        for step, heat in zip(steps, maps[(frame, layer, branch)]):
            local = heat[:, camera_offset:(camera_offset + 14)]
            mass = float(local.sum())
            merged = colorize(gt_view, local, scale, boxes)
            branch_name = {'video': 'Video', 'force': 'Force', 'action': 'AE'}[branch]
            row.append(panel(merged, f'{step} {branch_name} q / obs V | mass={mass:.3f}'))
        header_height = 24
        canvas = Image.new('RGB', (row[0].width * len(row), row[0].height + header_height), '#111111')
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width - 1, header_height - 1), fill='#252b32')
        draw.text((5, 4), f'{BRANCH_LABELS[branch]} | shared raw P 0..{scale:.3g}; sqrt display',
                  fill='#9eead8', font=ImageFont.load_default())
        for col, image in enumerate(row):
            canvas.paste(image, (col * image.width, header_height))
        rows.append(np.asarray(canvas))
    return np.concatenate(rows, axis=0)


def _normalized(vector):
    vector = np.asarray(vector, dtype=np.float64)
    vector = np.maximum(vector, 0.)
    total = vector.sum()
    return vector / total if total > 0 else np.full_like(vector, 1. / len(vector))


def _map_entropy(heat):
    p = _normalized(heat.ravel())
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    return entropy, float(np.exp(entropy))


def compose_query_frame(gt, query_maps, query_bias, baseline_force, posttrain_force,
                        boxes, episode, frame, layer, step, horizon, scales, view):
    from PIL import Image, ImageDraw, ImageFont
    camera_offset = 0 if view == 'global' else 14
    gt_view = gt[:, camera_offset * 16:(camera_offset + 14) * 16]
    mixture = query_maps.mean(0)
    prior = _normalized(np.exp(np.asarray(query_bias, dtype=np.float64)
                               - np.max(query_bias)))
    base_map = baseline_force[:392].reshape(14, 28)
    post_map = posttrain_force[:392].reshape(14, 28)
    panels = [
        ('GT', np.zeros((14, 28), dtype=np.float32), 'none'),
        ('Query mixture', mixture.reshape(14, 28), 'query'),
        ('Applied O-prior', prior.reshape(14, 28), 'prior'),
    ]
    panels.extend((f'Q{query:02d}', heat.reshape(14, 28), 'query')
                  for query, heat in enumerate(query_maps))
    panels.extend((('Force 20k', base_map, 'force'), ('Force 25k', post_map, 'force')))
    row = []
    for title, heat, scale_key in panels:
        local = heat[:, camera_offset:(camera_offset + 14)]
        mass = float(local.sum())
        scale = max(scales.get(scale_key, 0.), 1e-12)
        merged = colorize(gt_view, local, scale, boxes)
        suffix = '' if scale_key == 'none' else f' | view mass={mass:.3f}'
        row.append(panel(merged, f'{title}{suffix}'))
    header = 24
    canvas = Image.new('RGB', (row[0].width * len(row), row[0].height + header), '#111111')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width - 1, header - 1), fill='#252b32')
    draw.text((5, 4), f'Layer {layer} | {view} | Action h={horizon} -> Force h+1 | per-measure raw scales',
              fill='#9eead8', font=ImageFont.load_default())
    for col, image in enumerate(row):
        canvas.paste(image, (col * image.width, header))
    return np.asarray(canvas)


def write_video(path, frames, fps=1):
    import av
    first = next(frames)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = first.shape[1], first.shape[0]
        stream.pix_fmt = 'yuv420p'
        for image in (first, *frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    from PIL import Image
    from roi import region_statistics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=2, required=True,
                        help='baseline then later-checkpoint result directories; labels use protocol steps')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = [Path(x) for x in args.runs]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    protocols = load_runs(runs)
    episode = protocols[0]['video_episode']
    predictions = [np.load(run / 'predictions.npz') for run in runs]
    gt = np.load(runs[0] / f'episode_{episode:03d}_gt.npz')['rgb']
    if gt.shape[1:] != (224, 448, 3):
        raise ValueError(f'Unexpected dual-view frame layout: {gt.shape}')
    if not np.array_equal(np.load(runs[1] / f'episode_{episode:03d}_gt.npz')['rgb'], gt):
        raise ValueError('Recorded RGB differs between runs')

    probe_indices = protocols[0]['probe_indices']
    frames = []
    for sample in probe_indices:
        row = int(np.flatnonzero(predictions[0]['sample'] == sample)[0])
        if not all(int(np.flatnonzero(p['sample'] == sample)[0]) == row for p in predictions):
            raise ValueError('Probe-to-prediction row alignment differs')
        if int(predictions[0]['episode'][row]) != episode:
            raise ValueError('Probe is not from the selected video episode')
        frame = int(predictions[0]['frame'][row])
        if any(int(p['frame'][row]) != frame for p in predictions):
            raise ValueError('Probe frame differs between runs')
        frames.append(frame)

    roi = json.loads(Path(__file__).with_name('contact-roi.json').read_text())
    if roi['episode'] != episode:
        raise ValueError('ROI annotation belongs to another episode')
    maps = {}
    scales = {}
    roi_rows = []
    learned_horizons = {}
    learned_scales = {}
    learned_roi_rows = []
    learned_diversity_rows = []
    representative_horizons = {}
    steps = [p['step'] for p in protocols]
    for frame, sample in zip(frames, probe_indices):
        per_run = [np.load(run / f'attention_{sample:06d}.npz') for run in runs]
        row = int(np.flatnonzero(predictions[1]['sample'] == sample)[0])
        for layer in (13, 27):
            for branch in BRANCHES:
                key = f's15_l{layer:02d}_{branch}'
                pair = []
                for array in per_run:
                    probability = array[key].mean((0, 1))
                    if probability.shape[0] < 392:
                        raise ValueError(f'Missing observed video keys for {key}')
                    pair.append(probability[:392].reshape(14, 28))
                maps[(frame, layer, branch)] = pair
                scales[(layer, branch)] = max(scales.get((layer, branch), 0.0),
                                               *(float(m.max()) for m in pair))
                for step, heat in zip(steps, pair):
                    for view_index, view in enumerate(VIEWS):
                        stats = region_statistics(heat, roi['frames'][str(frame)][view], view_index)
                        roi_rows.append(dict(sample=sample, frame=frame, step=step,
                                             layer=layer, branch=branch, view=view, **stats))

            prefix = f's15_l{layer:02d}'
            query_key = prefix + '_query_maps'
            bias_key = prefix + '_query_bias'
            force_query_key = prefix + '_force_query_probs'
            if any(key not in per_run[1].files for key in (query_key, bias_key, force_query_key)):
                raise ValueError(f'Missing learned-query routing or per-horizon Force attention for {prefix}')
            base_force_key = prefix + '_force_query_probs'
            if base_force_key not in per_run[0].files:
                raise ValueError(f'Missing baseline per-horizon Force attention for {prefix}')
            raw_queries = per_run[1][query_key]
            raw_bias = per_run[1][bias_key]
            base_force = per_run[0][base_force_key]
            post_force = per_run[1][force_query_key]
            if (raw_queries.ndim != 4 or raw_queries.shape[0] != 1 or raw_queries.shape[1] != 32
                    or raw_queries.shape[-1] != 392 or raw_bias.shape != (1, 32, 392)
                    or base_force.ndim != 3 or post_force.ndim != 3
                    or base_force.shape[:2] != (1, 32) or post_force.shape[:2] != (1, 32)
                    or base_force.shape[-1] < 392 or post_force.shape[-1] < 392):
                raise ValueError(f'Unexpected query/bias/per-horizon attention shape for {prefix}')
            horizon_valid = predictions[1]['action_valid'][row].any(-1)
            if not horizon_valid.any():
                raise ValueError(f'No valid action horizons for query maps at frame {frame}')
            valid_ids = np.flatnonzero(horizon_valid)
            selected = sorted(set(int(valid_ids[round(i * (len(valid_ids) - 1) / 4)])
                                  for i in range(5)))
            representative_horizons[(frame, layer)] = selected
            for horizon in valid_ids:
                query_maps = raw_queries[0, horizon].astype(np.float64)
                mixture = query_maps.mean(0)
                prior = _normalized(np.exp(raw_bias[0, horizon].astype(np.float64)
                                           - np.max(raw_bias[0, horizon])))
                baseline_force = base_force[0, horizon].astype(np.float64)
                posttrain_force = post_force[0, horizon].astype(np.float64)
                learned_horizons[(frame, layer, int(horizon))] = dict(
                    query_maps=query_maps, query_bias=raw_bias[0, horizon],
                    baseline_force=baseline_force, posttrain_force=posttrain_force)
                measures = [('query_mixture', mixture), ('applied_prior', prior)]
                measures.extend((f'query_{q}', heat) for q, heat in enumerate(query_maps))
                measures.extend((('force_20k', baseline_force[:392]),
                                 ('force_25k', posttrain_force[:392])))
                for measure, values in measures:
                    heat = values.reshape(14, 28)
                    scale_kind = ('query' if measure.startswith('query_') else
                                  'prior' if measure == 'applied_prior' else
                                  'force' if measure.startswith('force_') else 'query')
                    skey = (layer, scale_kind)
                    learned_scales[skey] = max(learned_scales.get(skey, 0.), float(heat.max()))
                    entropy, support = _map_entropy(heat)
                    for view_index, view in enumerate(VIEWS):
                        stats = region_statistics(heat, roi['frames'][str(frame)][view], view_index)
                        learned_roi_rows.append(dict(sample=sample, frame=frame,
                            action_horizon=int(horizon), step=steps[1], layer=layer,
                            measure=measure, view=view, entropy=entropy,
                            effective_support=support, **stats))
                flat = query_maps.reshape(query_maps.shape[0], -1)
                norms = np.linalg.norm(flat, axis=1, keepdims=True).clip(min=1e-12)
                cosine = (flat / norms) @ (flat / norms).T
                upper = cosine[np.triu_indices(len(flat), k=1)]
                learned_diversity_rows.append(dict(sample=sample, frame=frame,
                    action_horizon=int(horizon), step=steps[1], layer=layer,
                    mean_pairwise_cosine=float(upper.mean()) if len(upper) else None,
                    query_count=int(len(flat))))

    records = []
    for layer in (13, 27):
        for view in VIEWS:
            view_frames = []
            sheets = []
            for frame in frames:
                boxes = roi['frames'][str(frame)][view]
                image = compose_frame(gt[frame], maps, boxes, episode, frame, frame / 30,
                                      layer, scales, steps, view)
                view_frames.append(image)
                sheets.append(Image.fromarray(image))
            video_path = out / f'episode_{episode:03d}_{view}_attention_s15_l{layer:02d}_comparison.mp4'
            write_video(video_path, iter(view_frames), fps=1)
            sheet_path = out / f'episode_{episode:03d}_{view}_attention_s15_l{layer:02d}_comparison.png'
            sheet = Image.new('RGB', (sheets[0].width, sum(x.height for x in sheets)), '#101418')
            y = 0
            for item in sheets:
                sheet.paste(item, (0, y)); y += item.height
            sheet.save(sheet_path)
            records.append(dict(layer=layer, view=view, video=video_path.name,
                                contact_sheet=sheet_path.name))

    learned_outputs = []
    for layer in (13, 27):
        for view in VIEWS:
            view_frames = []
            sheets = []
            for frame in frames:
                rows = []
                for horizon in representative_horizons[(frame, layer)]:
                    record = learned_horizons[(frame, layer, horizon)]
                    rows.append(compose_query_frame(gt[frame], record['query_maps'],
                        record['query_bias'], record['baseline_force'], record['posttrain_force'],
                        roi['frames'][str(frame)][view], episode, frame, layer, steps[1],
                        horizon, learned_scales, view))
                image = np.concatenate(rows, axis=0)
                view_frames.append(image)
                sheets.append(Image.fromarray(image))
            video_path = out / f'episode_{episode:03d}_{view}_visual_force_horizons_s15_l{layer:02d}_step{steps[1]:06d}.mp4'
            write_video(video_path, iter(view_frames), fps=1)
            sheet_path = out / f'episode_{episode:03d}_{view}_visual_force_horizons_s15_l{layer:02d}_step{steps[1]:06d}.png'
            sheet = Image.new('RGB', (sheets[0].width, sum(x.height for x in sheets)), '#101418')
            y = 0
            for item in sheets:
                sheet.paste(item, (0, y)); y += item.height
            sheet.save(sheet_path)
            learned_outputs.append(dict(layer=layer, view=view, video=video_path.name,
                                        contact_sheet=sheet_path.name))

    grouped = {}
    for row in roi_rows:
        grouped.setdefault((row['step'], row['layer'], row['branch'], row['view']), []).append(row)
    roi_summary = []
    for (step, layer, branch, view), rows in sorted(grouped.items()):
        summary = dict(step=step, layer=layer, branch=branch, view=view, anchors=len(rows))
        for field in ('view_attention_mass', 'region_attention_mass', 'area_fraction',
                      'within_view_fraction', 'enrichment'):
            values = [row[field] for row in rows if row[field] is not None]
            summary[field + '_mean'] = float(np.mean(values)) if values else None
            summary[field + '_median'] = float(np.median(values)) if values else None
        roi_summary.append(summary)
    (out / 'roi-summary.json').write_text(json.dumps(roi_summary, indent=2) + '\n')
    query_grouped = {}
    for row in learned_roi_rows:
        query_grouped.setdefault((row['step'], row['layer'], row['measure'], row['view']), []).append(row)
    query_summary = []
    for (step, layer, measure, view), rows in sorted(query_grouped.items()):
        summary = dict(step=step, layer=layer, measure=measure, view=view,
                       anchors=len({row['sample'] for row in rows}), action_horizon_observations=len(rows))
        for field in ('view_attention_mass', 'region_attention_mass', 'area_fraction',
                      'within_view_fraction', 'enrichment', 'entropy', 'effective_support'):
            values = [row[field] for row in rows if row[field] is not None]
            summary[field + '_mean'] = float(np.mean(values)) if values else None
            summary[field + '_median'] = float(np.median(values)) if values else None
        query_summary.append(summary)
    (out / 'query-force-roi-summary.json').write_text(json.dumps(query_summary, indent=2) + '\n')
    (out / 'query-force-roi-by-horizon.json').write_text(json.dumps(learned_roi_rows, indent=2) + '\n')
    (out / 'learned-query-diversity.json').write_text(json.dumps(learned_diversity_rows, indent=2) + '\n')
    manifest = dict(episode=episode, steps=steps, sampling_step=15,
                    layers=[13, 27], queries=list(BRANCH_LABELS),
                    probe_indices=probe_indices, frames=frames,
                    timestamps_seconds=[round(f / 30, 3) for f in frames],
                    timeline='Seven measured anchors only; no interpolation between attention probes.',
                    key_layout='first 392 visual keys are the observed 14x28 dual-view frame; each view is 14x14.',
                    color_scale='Fixed by layer/measure across all anchors, both views, and both checkpoints.',
                    representative_action_horizons='Five approximately evenly spaced valid horizons per anchor are visualized; all valid horizons are in the JSON statistics. Action horizon h maps to Force offset h+1.',
                    applied_prior='Softmax over the saved additive O-only log-bias; proportional to the exact multiplicative prior. It is normalized only over the 392 observed-frame keys for display.',
                    force_attention='Head-averaged per temporal query; includes the saved additive bias and original valid-key mask.',
                    display_transform=f'Probability colors and opacity use x^{DISPLAY_GAMMA}; shared raw probability scale is unchanged.',
                    scales={f'l{layer:02d}_{branch}': value for (layer, branch), value in scales.items()},
                    query_force_scales={f'l{layer:02d}_{measure}': value
                                        for (layer, measure), value in learned_scales.items()},
                    roi_summary='roi-summary.json',
                    query_force_roi_summary='query-force-roi-summary.json',
                    query_force_roi_by_horizon='query-force-roi-by-horizon.json',
                    learned_query_diversity='learned-query-diversity.json',
                    learned_query_outputs=learned_outputs, outputs=records)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
