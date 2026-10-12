# Architecture and experiment boundary

The user clarified that ForceWM stays in the Cosmos replacement experiment, and that AE/Force depth follows the official backbone rather than the PDF's14-layer proposal. The pinned official robot/action-cond has28 layers. The PDF's broader VLM reference planner, events and closed-loop physical interface are research context; they are not claimed as implemented in this backbone experiment.

## Initialization and native video

Official source: NVIDIA Cosmos-Predict2.5 commit `a2c298b0a3df3778b973fe65e9e58877b292d8a7`. Robot/action-cond weight revision `15a82a2ec231bc318692aa0456a36537c806e7d4`, file `robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt`. All expected video keys load strictly. Official tokenizer and Reason1 text weight shards have independent SHA checks. No Wan model weights initialize this experiment.

The native adapter preserves patching, native Q/K normalization,3D RoPE, global timestep scaling, three independent AdaLN branches, text projection, text cross-attention and output head. Single-device attention uses Torch SDPA, with differentiable Torch equivalents for Transformer Engine RMSNorm/RoPE; Accelerate owns four-rank data parallelism. Source files in the upstream submodule remain unchanged. This is a native-module adapter, not the upstream distributed training engine.

**Video keeps native bidirectional video attention and global diffusion time**, with the official action embedders adding per-latent AdaLN conditions. It does not read Force tokens. Supervised video uses clean teacher actions; free memory uses the current noisy/generated Action. There is no shared self-attention path from teacher video back to AE/Force. For policy observation, the same video parameters separately process only the clean current-frame latent at sigma0. Its K/V cannot depend on teacher future video. This avoids applying a new causal mask to the pretrained native video stream.

The requested FastWAM shifted-uniform FM recipe is retained: Video shift5, Action shift1, Gaussian starting noise,16 Euler inference steps. This differs from the full upstream Cosmos pretraining scheduler/trainer recipe. Preserve that distinction in comparisons.

## Visibility and gradients

Rows are query streams; columns are visible K/V or separate cross-attention memories.

| Query | Native video | Observed-only video | Action | HistoryH | CurrentActionUc | AR wrenchW | Free future video |
|---|---|---|---|---|---|---|---|
| Native video | Bidirectional within video | — | External controls through official AdaLN | No | No | No | — |
| Observation stream | No | Current frame only | No | No | No | No | No |
| AE | No teacher future | Same-layer current observationKV | BidirectionalAction | No | No | Free-AR memory only | Detached same-layer memory |
| H | No | Current image pooled condition | No | CausalH | No | No | No |
| Uc | No | Current image pooled condition | Encoded currentA_sigma+clock | No | BidirectionalUc | No | No |
| W | No | Current image pooled condition | ViaUc | All availableH | All availableUc | CausalW | Detached same-layer memory |

AE's generated-video and free-Force memories enter two independently zero-gated residual cross-attention blocks per layer (width512,8heads). Zero gates preserve the initial AE path; they are trainable. The primary AE MoT attention reads current-image K/V plus its own current noisy Action sequence. Observation K/V remains differentiable through the official video branch; generated future video and free AR are detached. Only supervised Force loss trains the Force branch directly.

Force history uses **H[t−31..t]** with state7+wrench6 and availability masks. Current Action uses offsets0..31, with its current FM clock. Force targets are **w[t+1..t+32]** reconstructed from immutable rawFP64 arrays. The cache's old wrench0..31 is never relabeled as future force. Teacher inputs are `[measured w0, true w1,...,true w31]`, with causalW attention. Free generation starts at measuredw0 and appends its own predictions; it accepts no force noise, future labels or teacher masks. Free video is generated from the current Action at every Action forward, then Force is refreshed from that Action/clock and the resulting video. At final generatedA0/clock0, both video and Force are regenerated together. No hold/reference/candidate trajectory initializes Action.

Generation sees the current image, text, current proprio, measured history and currentAction. Teacher future video and future wrench can affect only their supervised loss paths. Synthetic perturbation tests check AE output/loss independence from those labels. No future availability mask is used to shorten Force free generation.

## Action-to-video time mapping

The original dataset supplies z-score-normalized commands at offsets0..31; it has no additional delta transform. Nine RGB frames use offsets0,4,...,32. The official chunk model requires four7D controls per future latent, hence eight controls for the two future latents. We do not truncate to the first eight commands or average/sum unknown pose/rotation units.

A random shared `Linear(32,7)` maps each consecutive group of four7D commands plus four validity bits to one control. Controlj sees precisely commands[4j..4j+3] and conditions the interval between sampled frames4j and4j+4. All32 steps are represented, including the last interval28..31. Padded commands are zeroed before projection; an entirely padded interval has exactly zero control, including its bias. This projection is a learned dataset adapter, not physical equivalence to official robot actions. Its seed45 and mapping version are bound into the experiment manifest and checkpoint schema. The original two official action embedders remain strictly loaded and trainable.

Training: supervised noisy video + clean teacher controls → VideoFM; independent free video from first observation + detached currentA_sigma → AE/Force memories. Only the supervised video loss trains the control projection directly. Inference starts Action from Gaussian noise. Before each Action velocity, regenerate free video from currentAction, then freely roll out Force, then update Action. After the last Action step, regenerate video from finalA0 and refresh Force using that exact video. No teacher trajectory initializes this process. Current observationKV is computed separately without any action condition, retaining the safe attention separation.

Each free video generation costs16 denoising passes plus one feature pass. Training keeps17 generated +1 supervised +1 observed =19 full video passes per batch. Sixteen-step Action inference now requires17 such video generations =289 full passes plus one observation pass. These are counts, not measured throughput; no sampling steps are silently reduced.

## Feature/cache contract

The50 training episodes/34808 anchors and original preprocessing are unchanged. Reuse the validated FP32RGB cache, concatenate main-left/wrist-right, and gather all9 frames of each anchor before official Cosmos encoding. Independently encode the single observed image. Do not slice full-episode latents or concatenate independently encoded frames.

Cosmos features `[16,3,28,56]` cannot reuse Wan `[48,3,14,28]`. The new manifest binds RGB and numeric identities, FP64 Force sidecar identity, text/tokenizer/source identities, camera order and offsets. Shards are atomically completed and SHA checked. Training rejects partial caches, wrong episode/anchor order, missing FP64 future targets, wrong temporal masks and Wan features. It reads only prepared tensor shards plus task-text index; no image decoding or tokenizer is part of training.

## Validation limits

Local tests use small synthetic networks, including28-layer low-width networks, CPU forward/gradient parity and two actual CPU Adam updates. Local BF16 backward disables oneDNN on CPUs that lack its BF16 backward implementation. These results do not establish official-weight quality, B16 GPU memory, throughput or distributed persistence.

GPU acceptance requires current resources and complete actual features, all four ZeRO1 ranks atB16/global64, two Adam updates with finite moments/parameters, no skipped updates,28 Force layer calls, complete native checkpoint save and a fresh four-rank reload that matches every model/optimizer/RNG fingerprint. Launch gates also require unforced full process-tree closure with pidfd/start_ticks and restored subreaper. `process_ownership.py` includes the corrected PID replacement identity retirement logic. The user authorized no time limit for this Cosmos allocation; disposable probes remain bounded and formal execution stays in the foreground.

Training/evaluation are conditional fitting on the same50 episodes. Video consumes Action, but generated videos alone do not validate physical causality, closed-loop execution or generalization. The learned control mapping and robot/domain mismatch require real held-out evaluation. Force units/calibration and wrapped rotation metrics remain separate unverified dataset properties.
