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


def compose_query_frame(gt, query_maps, boxes, episode, frame, layer, step, vmax, view):
    from PIL import Image, ImageDraw, ImageFont
    camera_offset = 0 if view == 'global' else 14
    gt_view = gt[:, camera_offset * 16:(camera_offset + 14) * 16]
    row = [panel(colorize(gt_view, np.zeros((14, 14), dtype=np.float32), vmax, boxes),
                 f'Ep{episode:03d} f{frame} GT')]
    for query, heat in enumerate(query_maps):
        local = heat[:, camera_offset:(camera_offset + 14)]
        row.append(panel(colorize(gt_view, local, vmax, boxes),
                         f'{step} learned Q{query:02d} | view mass={float(local.sum()):.3f}'))
    header = 24
    canvas = Image.new('RGB', (row[0].width * len(row), row[0].height + header), '#111111')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width - 1, header - 1), fill='#252b32')
    draw.text((5, 4), f'Layer {layer}: learned action-conditioned query maps averaged over valid action horizons; shared raw P 0..{vmax:.3g}',
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
    learned_maps = {}
    learned_scales = {}
    learned_roi_rows = []
    learned_diversity_rows = []
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

            query_key = f's15_l{layer:02d}_query_maps'
            if query_key not in per_run[1].files:
                raise ValueError(f'Missing learned visual query maps: {query_key}')
            raw_queries = per_run[1][query_key]
            if raw_queries.ndim != 4 or raw_queries.shape[0] != 1 or raw_queries.shape[1] != 32 or raw_queries.shape[-1] != 392:
                raise ValueError(f'Unexpected learned query map shape for {query_key}: {raw_queries.shape}')
            horizon_valid = predictions[1]['action_valid'][row].any(-1)
            if not horizon_valid.any():
                raise ValueError(f'No valid action horizons for query maps at frame {frame}')
            valid_maps = raw_queries[0, horizon_valid].astype(np.float64)
            mean_queries = valid_maps.mean(0).reshape(raw_queries.shape[2], 14, 28)
            learned_maps[(frame, layer)] = mean_queries
            learned_scales[layer] = max(learned_scales.get(layer, 0.0), float(mean_queries.max()))
            flat = mean_queries.reshape(mean_queries.shape[0], -1)
            norms = np.linalg.norm(flat, axis=1, keepdims=True).clip(min=1e-12)
            cosine = (flat / norms) @ (flat / norms).T
            upper = cosine[np.triu_indices(len(flat), k=1)]
            learned_diversity_rows.append(dict(sample=sample, frame=frame, step=steps[1], layer=layer,
                mean_pairwise_cosine=float(upper.mean()) if len(upper) else None,
                query_count=int(len(flat))))
            for horizon, query_maps in zip(np.flatnonzero(horizon_valid), valid_maps):
                spatial = query_maps.reshape(raw_queries.shape[2], 14, 28)
                for query, heat in enumerate(spatial):
                    prob = np.maximum(heat, 1e-12)
                    prob = prob / prob.sum()
                    entropy = float(-(prob * np.log(prob)).sum())
                    for view_index, view in enumerate(VIEWS):
                        stats = region_statistics(heat, roi['frames'][str(frame)][view], view_index)
                        learned_roi_rows.append(dict(sample=sample, frame=frame, action_horizon=int(horizon),
                            step=steps[1], layer=layer, query=query, view=view,
                            entropy=entropy, effective_support=float(np.exp(entropy)), **stats))

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
                image = compose_query_frame(gt[frame], learned_maps[(frame, layer)],
                    roi['frames'][str(frame)][view], episode, frame, layer, steps[1],
                    learned_scales[layer], view)
                view_frames.append(image)
                sheets.append(Image.fromarray(image))
            video_path = out / f'episode_{episode:03d}_{view}_learned_queries_s15_l{layer:02d}_step{steps[1]:06d}.mp4'
            write_video(video_path, iter(view_frames), fps=1)
            sheet_path = out / f'episode_{episode:03d}_{view}_learned_queries_s15_l{layer:02d}_step{steps[1]:06d}.png'
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
        query_grouped.setdefault((row['step'], row['layer'], row['query'], row['view']), []).append(row)
    query_summary = []
    for (step, layer, query, view), rows in sorted(query_grouped.items()):
        summary = dict(step=step, layer=layer, query=query, view=view,
                       anchors=len({row['sample'] for row in rows}), action_horizon_observations=len(rows))
        for field in ('view_attention_mass', 'region_attention_mass', 'area_fraction',
                      'within_view_fraction', 'enrichment', 'entropy', 'effective_support'):
            values = [row[field] for row in rows if row[field] is not None]
            summary[field + '_mean'] = float(np.mean(values)) if values else None
            summary[field + '_median'] = float(np.median(values)) if values else None
        query_summary.append(summary)
    (out / 'learned-query-roi-summary.json').write_text(json.dumps(query_summary, indent=2) + '\n')
    (out / 'learned-query-diversity.json').write_text(json.dumps(learned_diversity_rows, indent=2) + '\n')
    manifest = dict(episode=episode, steps=steps, sampling_step=15,
                    layers=[13, 27], queries=list(BRANCH_LABELS),
                    probe_indices=probe_indices, frames=frames,
                    timestamps_seconds=[round(f / 30, 3) for f in frames],
                    timeline='Seven measured anchors only; no interpolation between attention probes.',
                    key_layout='first 392 visual keys are the observed 14x28 dual-view frame; each view is 14x14.',
                    color_scale='Fixed per layer and query branch across all anchors, both views, and both checkpoints.',
                    display_transform=f'Probability colors and opacity use x^{DISPLAY_GAMMA}; shared raw probability scale is unchanged.',
                    scales={f'l{layer:02d}_{branch}': value for (layer, branch), value in scales.items()},
                    roi_summary='roi-summary.json',
                    learned_query_summary='learned-query-roi-summary.json',
                    learned_query_diversity='learned-query-diversity.json',
                    learned_query_outputs=learned_outputs, outputs=records)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
