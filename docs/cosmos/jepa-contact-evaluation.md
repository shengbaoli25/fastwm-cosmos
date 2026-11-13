Plug insertion JEPA contact diagnostic
====================================

This evaluator loads the completed 30,000-step ViT-L JEPA model and its exact
resolved configuration. It never runs an optimizer, alters a training checkpoint,
or supplies future recorded actions/forces/images to the query. Proprio is sliced
to its current row before inference. The action query follows generated noisy
actions through the original 16-step joint sampler with fixed action, force and
video seeds.

The baseline model trained on all 78 trajectories. The default diagnostic uses
12 evenly spaced episode IDs and valid anchors at 20%, 50% and 80% of each
trajectory. These are temporal fractions, **not annotated contact phases**. This
is a descriptive mechanism check, not held-out evaluation or robot success.

Run with one visible GPU:

```bash
bash scripts/cosmos/launch_jepa_contact_eval.sh \
  --run-root /path/to/completed/plug/job/run \
  --output /path/to/fresh/evaluation \
  --episodes 12 --steps 16 --intervention-samples 12 \
  --roi /path/to/blinded-contact-roi.json
```

Use `--mode prepare` to produce the inventory, raw RGB sheets and ROI template
without constructing a model. Both observed frames are shown, in camera order
`cam_main`, `cam_wrist`. The official JEPA patch embed combines the past/current
pair (offsets -4 and 0, clamped within a continuous segment) into **one tubelet**.
The reader attends to 392 tokens: two 14x14 grids. A heatmap cannot attribute
attention independently to the two input frames.

The hooks record the exact query projection output and masked QK softmax used by
both visual reading layers, for every head/query/denoising call, without changing
the model forward result. Compressed arrays retain this full resolution. Preview
sheets average heads and queries; they show calls 0, midpoint and last, and use the
same red intensity scale of 0 to 10/392. They are previews, not the metric.

Annotate the visible plug/socket contact interface from raw RGB **before viewing
heatmaps**. Do not mark the whole gripper, socket or object as the contact surface.
Use normalized [x0,y0,x1,y1] boxes and explicitly mark occluded/ambiguous interfaces:

```json
{
  "schema": "jepa-contact-roi-v1",
  "annotation_stage": "raw_rgb_before_attention",
  "temporal_scope": "current_frame_only",
  "samples": {
    "123": {
      "cam_main": {"status": "visible", "boxes": [[0.4, 0.4, 0.5, 0.5]]},
      "cam_wrist": {"status": "occluded", "boxes": []}
    }
  }
}
```

Contact scores are withheld when annotations are absent. For visible interfaces,
the report gives within-camera attention mass, area-normalized enrichment
(uniform attention = 1), peak-point hit rate, and the camera's global attention
mass. Fractional patch/rectangle overlap is integrated exactly, including unions
of multiple boxes. Results are broken down by reader layer and camera. Bootstrap
intervals resample whole episodes, never treat heads, nearby frames or query slots
as independent demonstrations. Current-frame annotations still have the tubelet
temporal ambiguity above; a visible interface does not itself establish contact.

One anchor per selected episode also receives matched, fixed-seed interventions.
The 12 default intervention anchors alternate early/late across episodes; selection
is fixed before RGB, attention or annotations are examined. Conditions include masking
the top 20 attention patches per camera versus 20 random patches, setting trained
JEPA residual gates to zero, and a specified RGB channel scaling (1.15, .90, .80).
When an ROI is supplied, touched contact patches and a disjoint translation of
the **same complete mask** are masked as another pair. This matches patch count,
shape and spatial topology per camera. If a disjoint same-shape translation cannot
fit, that pair is skipped with an explicit reason. The separate top-attention
versus scattered-random comparison matches area but not necessarily topology;
it is exploratory and not the primary contact-specific contrast. Masking sets normalized RGB to
zero (mid-gray), in both observed frames. Only JEPA input changes; Cosmos observed
latents stay unchanged. The evaluator repeats the first baseline with identical
seeds and saves raw numerical differences. Force and Video differences under
interventions must stay below twice that measured repeatability (with a 1e-5
absolute floor); all raw differences remain in the metrics. This isolates this model's JEPA pathway, rather than testing
robustness of all vision pathways. The zero-gate condition is an ablation of the
same trained model, not an independently trained Cosmos baseline.

Report paired output changes and normalized action MSE changes. A changing action
does not automatically mean a worse action. One occlusion operator and at most 12 anchors
are exploratory evidence; they are not proof of semantic contact reasoning. For
stronger evidence, extend episode sampling, independently annotate unseen data,
repeat seeds, vary masking operators and run task-success evaluation.

After annotation, existing saved attention can be scored on CPU without another
GPU job:

```bash
bash scripts/cosmos/launch_jepa_contact_eval.sh --mode summarize \
  --output /path/to/evaluation --roi /path/to/blinded-contact-roi.json
```

Outputs include `run.json` with checkpoint/config/cache identity, `samples.json`,
`raw/`, `attention/`, `heatmaps/`, `predictions/`, `masks/`, `metrics.jsonl`,
`intervention-plan.json`, `progress.json`, and `report.json`. Exact patch masks,
per-camera counts, seed, fill value, observed frames and any skipped-pair reason
are saved. Annotation hashes are recorded when provided at execution. Separated,
occluded or ambiguous annotations never become contact metrics. The report records
whether annotations refer to the current frame or the union of both observed frames.
