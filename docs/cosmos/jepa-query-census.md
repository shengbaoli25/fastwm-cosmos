# JEPA plug insertion query census and local gallery

This is an exhaustive **inference diagnostic of the existing 30,000-step
checkpoint**. It does not train the proposed query loss. The target inventory is
78 demonstrations and 84,349 valid, continuity-filtered anchor windows. All were
used during training, so this is not a held-out performance evaluation.

## Output and resumability

The evaluator writes completed shards of up to 500 windows. A tar.gz shard holds
`attention.npy` with shape `[N,2,4,8,392]`, float16, averaged over the 16 denoising
steps; `samples.jsonl`; `metrics.jsonl`; and two current-frame JPEG thumbnails
per sample. `shard-manifest.json` records member hashes and sample IDs. Incomplete
archives are not gallery inputs. The original shards remain the inference source
of truth; the gallery only extracts and indexes them.

Two views each contribute a row-major 14×14 grid. Each token encodes a tubelet
of the past and current observed frame, clamped to the same continuous segment.
The current-frame thumbnail does not make the attention frame-specific. Averaging
over denoising steps removes information about step-dependent query changes.

## Build or update a partial gallery

```bash
python scripts/cosmos/build_jepa_query_gallery.py \
  --source /path/to/census-output \
  --output /path/to/query-gallery \
  --roi /path/to/frozen-blinded-interface-roi.provisional.json
```

Re-running adds completed shards and rebuilds the small index. Existing extracted
shards must retain the same archive hash. Changed or incomplete extraction fails
rather than silently mixing versions. No source artifact is changed or deleted.
The coverage banner distinguishes partial results from the complete contiguous
84,349-row inventory. Missing shards remain missing; no synthetic windows appear.

The earlier 36-sample diagnostic can populate a working preview without running
the model again:

```bash
python scripts/cosmos/build_jepa_query_gallery.py \
  --source /path/to/old-36-sample-output --legacy36 \
  --output /path/to/query-gallery-preview \
  --roi /path/to/frozen-blinded-interface-roi.provisional.json
```

Use a separate output folder for a legacy preview and a census. The preview is
explicitly labeled 36 samples, partial coverage. It does not count as the census.

## Open locally

Run the generated `serve-gallery.cmd` on this Windows host, or:

```bash
python /path/to/query-gallery/serve_gallery.py --port 8765
```

Open `http://127.0.0.1:8765/`. The standard-library server binds only to localhost;
there is no publication, upload, third-party JavaScript, CDN, or external request.
Ctrl+C stops it. A plain file URL cannot fetch the binary shards, so the page
explains the server command when opened that way.

The custom server implements single HTTP byte ranges. One selected attention
window reads 50,176 bytes from its `.npy` shard instead of downloading a 25 MB
500-window array. A small cache keeps eight decoded windows and four metadata
shards. Thumbnails and row metadata are loaded on demand. The compact overview
contains only IDs and four scalar descriptors, allowing episode filters, ordered
timelines, paging, direct ID lookup, and global descriptor sorting without loading
all attention arrays.

## Interpretation

The default order is the original sample order, not a visually favorable example.
Users can choose layer, head and query or their means. Heatmaps use the exact
14×14 patch grid with a shared, explicit color scale. The default displays the
within-view probability relative to a uniform grid. The global probability view
retains the two-camera mass split. The float16 row is renormalized for display
only; its original probability sum is reported. Thumbnails are JPEG previews;
numerical attention is never computed from those compressed images.

The overview's entropy and peak are computed per layer/head/query and then
averaged. Query redundancy is the mean of 28 pairwise query cosine similarities
within each layer/head. In the census these descriptors are computed in float32
before storing float16 attention. Low entropy, high peak or low redundancy may
help find visually different candidates; none is labeled “good contact”, contact
accuracy, success, or a learned force-grounding quality score.

The optional frozen annotation file refers only to previously inspected IDs.
Unannotated windows remain unannotated. Visible boxes are provisional projected
plug/socket interface proxies, not measured force-contact locations. Occluded,
separated, and ambiguous views are excluded from interface enrichment reporting.
No boxes may be newly drawn after looking at favorable heatmaps and then used as
unbiased evaluation labels. The fixed main-camera socket geometry is a possible
spatial-prior confound. Robust claims require independent annotations, matched
interventions, and genuinely unseen demonstrations or layouts.

## Practical size

At 84,349 windows, float16 mean attention is about 4.23 GB before archive
compression. Thumbnails are additional. The gallery is deliberately sharded;
there is no multi-gigabyte single HTML file. Keep the source archives and gallery
directory on a disk with enough free space for both if extracting a full copy.
