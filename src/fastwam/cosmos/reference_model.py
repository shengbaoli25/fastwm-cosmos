"""Observation-audited plan conditioning for the unchanged V/F/A flow outputs."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .noaction import NoActionJointModel, BRANCH_READS, frame_slice
from .model import temporal_mse
from .reference_auditor import ObservationReferenceAuditor, alignment_loss
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d

SCHEMA = 'cosmos-reference-audited-vfa-joint8-v1'


class ReferenceJointModel(NoActionJointModel):
    checkpoint_schema = SCHEMA
    reference_mode = 'obs_only_full_plan_soft_local32'

    def __init__(self, net, *, reference_hidden=128, reference_heads=4, reference_layers=2,
                 reference_seed=46, alignment_weight=.1, **kwargs):
        super().__init__(net, **kwargs)
        if not 0 <= alignment_weight <= 10:
            raise ValueError('Finite nonnegative bounded alignment loss weight required')
        self.alignment_weight = float(alignment_weight)
        # The stock trainer optimizes dit + proprio_encoder, so all new weights
        # live in dit and are covered by optimizer/full-state checkpointing.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(reference_seed)
            self.dit['reference_auditor'] = ObservationReferenceAuditor(hidden=reference_hidden,
                heads=reference_heads, layers=reference_layers, state_dim=self.state_dim)
            self.dit['reference_condition'] = nn.Sequential(nn.Linear(reference_hidden,self.dit['action'].hidden_dim),
                nn.SiLU(), nn.Linear(self.dit['action'].hidden_dim,self.dit['action'].hidden_dim,bias=False))
        self.manifest.update(schema=SCHEMA, joint_protocol=SCHEMA, reference_mode=self.reference_mode,
            reference=dict(hidden=reference_hidden,heads=reference_heads,layers=reference_layers,seed=reference_seed,
                full_plan_length='variable',local_condition_tokens=32,condition_destination='AE input tokens only',
                observations='independent current frame latent, current proprio/TCP, causal masked history',
                selection='observation progress attention and learnt soft lookahead in augmented plan order',
                alignment_weight=self.alignment_weight,teacher_alignment_input=False),
            loss='original video/force/action FM plus masked provenance alignment CE')
        self.dit['runtime'].bind_manifest(self.manifest)

    def _conditions(self, sample):
        base = super()._conditions(sample)
        audit = self.dit['reference_auditor'](sample['reference_inputs'],
            first_frame_latents=sample['first_frame_latents'],proprio=sample['proprio'],
            history=sample['history'],history_is_pad=sample['history_is_pad'],
            history_channel_valid=sample['history_channel_valid'])
        condition = self.dit['reference_condition'](audit['local_tokens'])
        # The projection has a hidden bias. Explicit presence gating guarantees
        # whole-plan dropout also removes that bias, restoring observation-only AE.
        condition = condition * audit['reference_present'][:,None,None].to(condition.dtype)
        return base, condition, audit

    def prefill_static(self, first, conditions):
        # Only original causal O/H and text are cached. Reference never reaches
        # Video/Force; the AE condition is independent of diffusion iteration.
        return super().prefill_static(first, conditions[0])

    def joint_denoise_core(self, video, force, action, tv, tf, ta, conditions, static_cache=None):
        base, reference_condition, _ = conditions
        raw, context, cmask, history, hvalid = base
        full = self._video_prepare(video, tv, raw)
        op = frame_slice(full, 0, 1); vp = frame_slice(full, 1, full['grid'][0])
        observed, vt = op['tokens'], vp['tokens']
        ft, fm = self.dit['force'].prepare_future(force, tf, tv, context)
        at, _, am, text, text_mask, freq = self.dit['action'].prepare(action, ta, context, cmask)
        if at.shape != reference_condition.shape:
            raise ValueError('Exactly 32 local reference tokens must match the 32 AE input slots')
        at = at + reference_condition
        source_time = torch.cat((sinusoidal_embedding_1d(256, tv), sinusoidal_embedding_1d(256, tf)), -1)
        am = am + self.dit['action_source_time'](source_time).unflatten(-1, (6, self.dit['action'].hidden_dim))
        if static_cache is not None and (torch.is_grad_enabled() or len(static_cache['layers']) != self.depth):
            raise ValueError('Request-local inference cache with matching depth required')
        for i in range(self.depth):
            def layer(o, h, v, f, a, i=i):
                if static_cache is None:
                    o, h, kv = self._static_layer(i, o, h, op, hvalid)
                else:
                    kv = static_cache['layers'][i]
                v, f, a = self._dynamic_layer(i, v, f, a, vp, fm, am, text, text_mask, freq, hvalid, kv,
                    None if static_cache is None else static_cache['text'][i])
                return o, h, v, f, a
            args = (observed, history, vt, ft, at)
            observed, history, vt, ft, at = (checkpoint(layer, *args, use_reentrant=False)
                if self.gradient_checkpointing and torch.is_grad_enabled() else layer(*args))
        future_velocity = self.video.finish(vt, vp)
        vv = torch.cat((torch.zeros_like(video[:, :, :1]), future_velocity), 2)
        return vv, self.dit['force'].head(ft), self.dit['action'].post(at)

    def training_loss(self, sample):
        # Same clocks, noise distributions, action targets, masks and FM weighting
        # as NoActionJointModel. Auditor executes once, not an action prepass.
        clean = sample['video_latents'].to(self.device, self.torch_dtype)
        first = sample['first_frame_latents'].to(self.device, self.torch_dtype)
        action = sample['action'].to(self.device, self.torch_dtype)
        force = sample['wrench_future'].to(self.device, self.torch_dtype)
        if clean.shape[1:3] != (16,3) or first.shape != (*clean.shape[:2],1,*clean.shape[3:]) or action.shape[1:] != (32,self.action_dim) or force.shape[1:] != (32,6):
            raise ValueError('Expected original Video16x3, independent first frame, Action32, Force32x6')
        conditions = self._conditions(sample)
        tv, tf, ta = (s.sample_training_t(len(clean), self.device, self.torch_dtype)
                      for s in (self.v_scheduler, self.f_scheduler, self.a_scheduler))
        vn, fn, an = torch.randn_like(clean), torch.randn_like(force), torch.randn_like(action)
        fvalid = sample['wrench_valid'].to(self.device, torch.bool) & ~sample['force_is_pad'].to(self.device, torch.bool)[...,None]
        force = torch.where(fvalid, force, 0)
        v = self.v_scheduler.add_noise(clean, vn, tv); v[:,:, :1] = first
        f = self.f_scheduler.add_noise(force, fn, tf)
        a = self.a_scheduler.add_noise(action, an, ta)
        pv, pf, pa = self.joint_denoise_core(v,f,a,tv,tf,ta,conditions)
        imagepad = sample['image_is_pad'].to(self.device, torch.bool)
        vvalid = ~imagepad[:,1:].reshape(len(clean),2,4).all(-1)
        lv = temporal_mse(pv[:,:,1:].transpose(1,2).flatten(2),
            self.v_scheduler.training_target(clean,vn,tv)[:,:,1:].transpose(1,2).flatten(2),vvalid,self.v_scheduler,tv)
        la = temporal_mse(pa,self.a_scheduler.training_target(action,an,ta),
            ~sample['action_is_pad'].to(self.device,torch.bool),self.a_scheduler,ta)
        delta = torch.where(fvalid,pf.float()-self.f_scheduler.training_target(force,fn,tf).float(),0)
        lf = ((delta.square().sum((1,2))/fvalid.sum((1,2)).clamp_min(1))*self.f_scheduler.training_weight(tf)).mean()
        supervision = sample['reference_supervision']
        laudit = alignment_loss(conditions[2]['alignment_log_probabilities'],
            supervision['alignment_target'],supervision['alignment_enabled'])
        return lv+lf+la+self.alignment_weight*laudit, dict(loss_video_fm=float(lv.detach()),
            loss_force_fm=float(lf.detach()),loss_action_fm=float(la.detach()),loss_reference_alignment=float(laudit.detach()),
            reference_present_fraction=float(conditions[2]['reference_present'].float().mean()),
            reference_lookahead_span=float(conditions[2]['span'].detach().mean()))


def create_reference_joint_cosmos(*, upstream, checkpoint, text_cache, expected_text_identity,
        expected_data_identity, expected_reference_identity, expected_normalization_sha256=None,
        training_recipe=None, device='cuda', **kwargs):
    from .assets import (read_text_bank,load_nonaction_video,UPSTREAM_COMMIT,MODEL_ID,REVISION,
                         NONACTION_WEIGHT_SHA,TOKENIZER_SHA)
    from .joint_action import joint_action_contract
    representation = kwargs.get('action_representation','absolute')
    joint_action_contract(representation)
    if representation != 'delta_current_joint7' or kwargs.get('action_space','joint8') != 'joint8':
        raise ValueError('Reference experiment requires explicit current-anchor Joint8 deltas')
    if any(not isinstance(x,str) or len(x)!=64 for x in
           (expected_text_identity,expected_data_identity,expected_reference_identity,expected_normalization_sha256)):
        raise ValueError('Pinned data/text/reference/normalization identities required')
    bank, _, text_identity = read_text_bank(text_cache)
    if text_identity != expected_text_identity:
        raise ValueError('Pinned text cache identity differs')
    net = load_nonaction_video(upstream,checkpoint,device='cpu')
    recipe=dict(batch_per_rank=64,max_steps=30000)
    recipe.update(training_recipe or {})
    recipe.update(ranks=4,accum=1,global_batch=4*int(recipe['batch_per_rank']))
    manifest = dict(upstream=UPSTREAM_COMMIT,model=MODEL_ID,revision=REVISION,
        video_variant='base/post-trained',video_sha256=NONACTION_WEIGHT_SHA,dataset_identity=expected_data_identity,
        reference_identity=expected_reference_identity,normalization_sha256=expected_normalization_sha256,
        tokenizer_sha256=TOKENIZER_SHA,text_identity=text_identity,extra_config=kwargs,recipe=recipe)
    return ReferenceJointModel(net,manifest=manifest,text_bank=bank,**kwargs).to(device=device,dtype=torch.bfloat16)
