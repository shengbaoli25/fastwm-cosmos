"""Finish plots from fully saved predictions after a plotting-only failure."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from evaluate import fit_summary, plot_episode, dump


def validate(out):
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol.get('limited_smoke') or protocol.get('status')!='running':
        raise ValueError('Only a full interrupted evaluation can be finalized')
    values=np.load(out/'predictions.npz')
    n=len(values['sample'])
    if n!=protocol['windows'] or len(set(values['sample'].tolist()))!=n:
        raise ValueError('Full unique predicted windows required')
    if len(set(values['episode'].tolist()))!=protocol['episodes']:
        raise ValueError('Missing evaluated episodes')
    rows=[{key:values[key][i] for key in values.files} for i in range(n)]
    actual=fit_summary(rows);saved=json.loads((out/'metrics.json').read_text())
    for field in ('action','force','action_hold','force_hold'):
        for metric in ('mae','rmse','count'):
            np.testing.assert_allclose(actual[field][metric],saved[field][metric],rtol=1e-12,atol=1e-12)
    errors=saved['attention_output_parity']
    if sorted(e['sample'] for e in errors)!=protocol['probe_indices'] or any(any(x!=0 for x in e['max_abs'].values()) for e in errors):
        raise ValueError('All attention output parity receipts required')
    keys={f's{s:02d}_l{l:02d}_{branch}' for s in protocol['attention_steps']
          for l in protocol['attention_layers'] for branch in ('video','force','action')}
    for j in protocol['probe_indices']:
        probes=np.load(out/f'attention_{j:06d}.npz')
        if set(probes.files)!=keys:raise ValueError('Missing attention maps')
        for key in probes.files:
            if not np.isfinite(probes[key]).all():raise ValueError('Nonfinite attention')
            np.testing.assert_allclose(probes[key].sum(-1),1,atol=1e-5)
    ep=protocol['video_episode']
    gt=np.load(out/f'episode_{ep:03d}_gt.npz')['rgb']
    video=np.load(out/'video_predictions.npz');vm=json.loads((out/'video_metrics.json').read_text())
    frames=video['frames']
    if video['rgb'].shape!=(len(frames),224,448,3) or gt.shape[1:]!=(224,448,3):
        raise ValueError('Incomplete video shape')
    if not len(frames) or np.any(np.diff(frames)<=0) or frames.min()<0 or frames.max()>=len(gt):
        raise ValueError('Invalid predicted video timestamps')
    pairs={(e['frame'],e['view']) for e in vm['per_frame']}
    if pairs!={(int(f),v) for f in frames for v in (0,1)} or len(vm['per_frame'])!=len(pairs):
        raise ValueError('Video score receipt differs from decoded frames')
    return protocol,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--skip-plots',action='store_true',help='Validate all saved predictions and defer plots to local CPU')
    a=p.parse_args();out=Path(a.output)
    protocol,rows=validate(out)
    if not a.skip_plots:
        for ep in sorted(set(int(r['episode']) for r in rows)):
            plot_episode(out,ep,[r for r in rows if r['episode']==ep])
    protocol.update(status='complete',completed_windows=len(rows),ended_unix=time.time(),
                    plots_deferred=a.skip_plots,
                    cpu_postprocessing_recovery='matplotlib missing after all predictions and video decoding saved')
    dump(out/'protocol.json',protocol)
    print('CPU_POSTPROCESS_RECOVERED',len(rows),flush=True)


if __name__=='__main__':main()
