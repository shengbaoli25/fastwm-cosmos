"""Cosmos28 + random Action28 + direct supervised Force28 chunk.

Native global-time bidirectional video; isolated current-image observation KV.
Both world models use explicit clean actions; future video/force labels stay in losses.
Action FM is conditional reconstruction when its WM controls are target actions.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import flash_attention
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.forcewm.model import ActionDualMemory, masked_force_mse
from fastwam.forcewm.chunk import ChunkForceWM
from fastwam.baseline.optimizer import bounded_optimizer_param_groups
from .expert import CosmosExpert
from .state import ForceRuntimeState
from .controls import VideoControlAdapter


class _ActionOps:
    _split_modulation = staticmethod(MoT._split_modulation)
    _build_expert_attention_io = MoT._build_expert_attention_io
    _apply_expert_post_block_tensor = staticmethod(MoT._apply_expert_post_block_tensor)


def temporal_mse(pred, target, valid, scheduler, clock):
    if pred.shape != target.shape or valid.shape != pred.shape[:2]:
        raise ValueError('Loss shape/validity mismatch')
    delta = torch.where(valid[..., None], pred.float() - target.float(), 0)
    value = delta.square().mean(-1).sum(-1) / valid.sum(-1).clamp_min(1)
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite loss')
    return (value * scheduler.training_weight(clock)).mean()


class CosmosForceWM(nn.Module):
    force_type = ChunkForceWM
    checkpoint_schema = 'fastwam-cosmos-clean-action-force-chunk28-v3'

    def __init__(self, net, *, manifest, text_bank=None, action_hidden=1024,
                 action_ffn=4096, force_hidden=1280, force_ffn=4608,
                 force_heads=16, memory_width=512, memory_heads=8,
                 action_seed=42, force_seed=43, memory_seed=44, control_seed=45,
                 generated_video_steps=16, gradient_checkpointing=True, fps=7.5):
        super().__init__()
        self.manifest = dict(manifest)
        self.gradient_checkpointing = gradient_checkpointing
        self.generated_video_steps, self.fps = int(generated_video_steps), float(fps)
        if self.generated_video_steps < 1:
            raise ValueError('Generated Video step count must be positive')
        video = CosmosExpert(net)
        self.depth = len(net.blocks)
        text_dim = net.blocks[0].cross_attn.k_proj.in_features
        acfg = dict(action_dim=7, hidden_dim=action_hidden, ffn_dim=action_ffn,
                    num_heads=video.num_heads, attn_head_dim=video.attn_head_dim,
                    num_layers=self.depth, text_dim=text_dim, freq_dim=256,
                    eps=1e-6, use_gradient_checkpointing=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_seed)
            action = ActionDiT(**acfg)
            self.proprio_encoder = nn.Linear(8, text_dim)
            torch.manual_seed(force_seed)
            force = self.force_type(video_hidden_dim=video.hidden_dim,
                text_dim=text_dim, hidden_dim=force_hidden, ffn_dim=force_ffn,
                num_heads=force_heads, num_layers=self.depth, observation_channels=16,
                checkpoint_layers=gradient_checkpointing)
            memories = nn.ModuleList([ActionDualMemory(acfg, video.hidden_dim,
                force_hidden, memory_width, memory_heads) for _ in range(self.depth)])
            torch.manual_seed(control_seed)
            controls = VideoControlAdapter()
        # Original trainer optimizes .dit and .proprio_encoder, saves all ranks.
        self.dit = nn.ModuleDict(dict(video=video, action=action, force=force,
            memories=memories, controls=controls, runtime=ForceRuntimeState(memory_seed, self.manifest)))
        self.register_buffer('text_bank', text_bank, persistent=False)
        self.v_scheduler = WanContinuousFlowMatchScheduler(shift=5.)
        self.a_scheduler = WanContinuousFlowMatchScheduler(shift=1.)
        self.compile_training_denoise = False
        self.ops = _ActionOps()

    @property
    def device(self): return next(self.parameters()).device
    @property
    def torch_dtype(self): return self.dit['action'].head.weight.dtype
    @property
    def video(self): return self.dit['video']

    def _apply(self, fn):
        result = super()._apply(fn)
        if hasattr(self, 'dit') and self.video.net.use_wan_fp32_strategy:
            from .native import preserve_fp32_strategy
            preserve_fp32_strategy(self.video.net)
        return result

    def optimizer_param_groups(self, parameters):
        return bounded_optimizer_param_groups(self, parameters)

    def _contexts(self, sample):
        if 'cosmos_context' in sample:  # Also used by isolated CPU contract tests.
            raw = sample['cosmos_context'].to(self.device, self.torch_dtype)
        else:
            if self.text_bank is None: raise ValueError('Verified Cosmos Reason1 text bank required')
            ids = sample['cosmos_text_index'].to(self.device, torch.long).reshape(-1)
            if bool((ids < 0).any() | (ids >= len(self.text_bank)).any()):
                raise ValueError('Text index outside pinned bank')
            raw = (self.text_bank[ids[0]].unsqueeze(0)
                   if bool((ids == ids[0]).all()) else self.text_bank[ids])
        text = self.video.net.crossattn_proj(raw) if self.video.net.use_crossattn_projection else raw
        text = text.expand(sample['proprio'].shape[0], -1, -1)
        proprio = sample['proprio'].to(self.device, self.torch_dtype)
        if proprio.ndim == 3: proprio = proprio[:, 0]
        if proprio.shape != (text.shape[0], 8): raise ValueError('Current proprio8 required')
        context = torch.cat((text, self.proprio_encoder(proprio)[:, None]), 1)
        return text, context, torch.ones(context.shape[:2], device=self.device, dtype=torch.bool)

    def _video(self, latents, clock, raw_context, *, controls=None, collect=False, observed=False):
        prepared = self.video.prepare(latents, clock.float()/1000., raw_context,
            candidate_actions=controls,
            observed_only=observed, fps=self.fps, context_is_projected=True,
            observed_time='clean_zero' if observed else 'native_global')
        x = prepared['tokens']; layers=[]; kv=[]
        for i in range(self.depth):
            def run(tokens, i=i):
                q, k, v, state = self.video.block_qkv(i, tokens, prepared)
                out = self.video.block_finish(i, F.scaled_dot_product_attention(q,k,v), state, prepared)
                return out, k.transpose(1,2).flatten(2), v.transpose(1,2).flatten(2)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x,k,v = checkpoint(run,x,use_reentrant=False)
            else: x,k,v = run(x)
            if observed: kv.append((k,v))
            if collect: layers.append(x[:,prepared['tokens_per_frame']:].detach())
        return self.video.finish(x,prepared), tuple(layers), tuple(kv)

    @torch.no_grad()
    def generate_video_memory(self, first, raw_context, action_condition, action_is_pad):
        # Explicit clean controls; latent diffusion noise remains video-only.
        controls = self.dit['controls'](action_condition.detach(), action_is_pad).detach()
        shape = (*first.shape[:2], 3, *first.shape[3:])
        x = self.dit['runtime'].normal(shape).to(self.device,self.torch_dtype)
        x[:,:,:1] = first.detach()
        times,deltas = self.v_scheduler.build_inference_schedule(self.generated_video_steps,self.device,x.dtype)
        for t,dt in zip(times,deltas):
            v,_,_ = self._video(x,t.expand(len(x)),raw_context,controls=controls)
            x = self.v_scheduler.step(v,dt,x); x[:,:,:1] = first.detach()
        _,layers,_ = self._video(x,torch.zeros(len(x),device=self.device,dtype=x.dtype),raw_context,controls=controls,collect=True)
        return x.detach(), layers, torch.zeros(layers[0].shape[:2],device=self.device,dtype=torch.bool)

    def _condition(self, sample, action_condition, context, cmask, memory):
        x,layers,pad = memory
        return dict(history=sample['history'].to(self.device,self.torch_dtype),
            action_condition=action_condition.detach(),context=context.detach(),context_mask=cmask,
            history_is_pad=sample['history_is_pad'].to(self.device,torch.bool),
            history_channel_valid=sample['history_channel_valid'].to(self.device,torch.bool),
            candidate_is_pad=sample['action_is_pad'].to(self.device,torch.bool),
            # Native Cosmos velocity/scheduler may retain FP32. Force weights
            # use the expert dtype even when DeepSpeed disables autocast.
            observation_latents=x[:,:,:1].to(self.torch_dtype),video_features=layers,video_is_pad=pad)

    def _supervised_video(self, sample, noisy, clock, raw, context, cmask, action, apad):
        return self._video(noisy, clock, raw, controls=self.dit['controls'](action, apad))[0]

    def _world_models(self, sample, first, raw, context, cmask, action, apad):
        memory = self.generate_video_memory(first, raw.detach(), action, apad)
        force = self.dit['force'](**self._condition(sample, action, context, cmask, memory))
        return memory, force

    def action_velocity(self, action, clock, context, cmask, observation_kv, memory, force, apad):
        # AE diffusion states never flow back into either world model.
        expert = self.dit['action']
        x,_,mod,text,mask,freq = expert.prepare(action,clock,context,cmask)
        for i,(kobs,vobs) in enumerate(observation_kv):
            block=expert.blocks[i]; mem=self.dit['memories'][i]
            def run(tokens,kobs,vobs,block=block,mem=mem,i=i):
                q,k,v,res,gate,shift,scale,mg,_=self.ops._build_expert_attention_io(expert,block,tokens,freq,mod)
                allowed=torch.cat((torch.ones(len(tokens),kobs.shape[1],device=self.device,dtype=torch.bool),~apad),1)[:,None,None,:]
                mixed=flash_attention(q,torch.cat((kobs,k),1),torch.cat((vobs,v),1),expert.num_heads,ctx_mask=allowed)
                out=self.ops._apply_expert_post_block_tensor(block,res,mixed,gate,shift,scale,mg,text,mask)
                return mem(out,memory[1][i],force.layers[i],memory[2],force.is_pad,apad)
            x=checkpoint(run,x,kobs,vobs,use_reentrant=False) if torch.is_grad_enabled() and self.gradient_checkpointing else run(x,kobs,vobs)
        return expert.post(x), force

    def training_loss(self, sample):
        if 'image' in sample or 'video_latents' not in sample or 'first_frame_latents' not in sample:
            raise ValueError('Training requires independent Cosmos features; RGB/online VAE forbidden')
        clean=sample['video_latents'].to(self.device,self.torch_dtype)
        first=sample['first_frame_latents'].to(self.device,self.torch_dtype)
        action=sample['action'].to(self.device,self.torch_dtype)
        if clean.shape[1:3]!=(16,3) or first.shape!=(*clean.shape[:2],1,*clean.shape[3:]) or action.shape[1:]!=(32,7):
            raise ValueError('Cosmos16x3 video, current16x1 and Action32x7 required')
        raw,ctx,cmask=self._contexts(sample)
        vt=self.v_scheduler.sample_training_t(len(clean),self.device,self.torch_dtype)
        at=self.a_scheduler.sample_training_t(len(clean),self.device,self.torch_dtype)
        vn=torch.randn_like(clean); an=torch.randn_like(action)
        noisy=self.v_scheduler.add_noise(clean,vn,vt); noisy[:,:,:1]=first
        na=self.a_scheduler.add_noise(action,an,at)
        apad=sample['action_is_pad'].to(self.device,torch.bool)
        # The same clean GT action conditions supervised and generated video.
        pred=self._supervised_video(sample,noisy,vt,raw,ctx,cmask,action,apad)
        # Future latent covers four original frames. Partially valid group is kept.
        imagepad=sample['image_is_pad'].to(self.device,torch.bool)
        vvalid=~imagepad[:,1:].reshape(len(clean),2,4).all(-1)
        target=self.v_scheduler.training_target(clean,vn,vt)
        lv=temporal_mse(pred[:,:,1:].transpose(1,2).flatten(2),target[:,:,1:].transpose(1,2).flatten(2),vvalid,self.v_scheduler,vt)
        memory,force=self._world_models(sample,first,raw,ctx,cmask,action,apad)
        _,_,kv=self._video(first,torch.zeros_like(vt),raw,observed=True)
        pa,_=self.action_velocity(na,at,ctx,cmask,kv,memory,force,apad)
        la=temporal_mse(pa,self.a_scheduler.training_target(action,an,at),~sample['action_is_pad'].to(self.device,torch.bool),self.a_scheduler,at)
        gt=sample['wrench_future'].to(self.device,self.torch_dtype)
        fpad=sample['force_is_pad'].to(self.device,torch.bool); fvalid=sample['wrench_valid'].to(self.device,torch.bool)
        lf=masked_force_mse(force.wrench,gt,fpad,fvalid)
        return lv+la+lf, dict(loss_video=float(lv.detach()),loss_action=float(la.detach()),loss_force=float(lf.detach()))

    def forward(self,sample): return self.training_loss(sample)

    @torch.no_grad()
    def infer_forcewm(self, sample, *, action_condition, num_inference_steps=16,
                      seed=0, video_seed=44, generate_action=True):
        """Conditional WM evaluation; AE output is conditional reconstruction.

        action_condition is mandatory and already normalized, clean [B,32,7].
        No fallback to sample action or intermediate AE noise is permitted.
        video_seed is independent of the AE sampling seed.
        """
        prior_modes={module:module.training for module in self.modules()}; self.eval()
        prior_rng=self.dit['runtime'].get_state()
        self.dit['runtime'].manual_seed(video_seed)
        try:
            first=sample['first_frame_latents'].to(self.device,self.torch_dtype)
            if first.shape[1:3]!=(16,1): raise ValueError('Only observed Cosmos latent accepted')
            controls=action_condition.to(self.device,self.torch_dtype)
            if controls.shape != (len(first),32,7): raise ValueError('Explicit clean Action32x7 required')
            raw,ctx,cmask=self._contexts(sample)
            apad=sample['action_is_pad'].to(self.device,torch.bool)
            memory,force=self._world_models(sample,first,raw,ctx,cmask,controls,apad)
            result=dict(force=force.wrench,video_latents=memory[0])
            if generate_action:
                if num_inference_steps < 1: raise ValueError('Positive Action generation steps required')
                zero=torch.zeros(len(first),device=self.device,dtype=self.torch_dtype)
                _,_,kv=self._video(first,zero,raw,observed=True)
                gen=torch.Generator(device='cpu').manual_seed(seed)
                action=torch.randn((len(first),32,7),generator=gen).to(self.device,self.torch_dtype)
                times,deltas=self.a_scheduler.build_inference_schedule(num_inference_steps,self.device,self.torch_dtype)
                for t,dt in zip(times,deltas):
                    velocity,_=self.action_velocity(action,t.expand(len(first)),ctx,cmask,kv,memory,force,apad)
                    action=self.a_scheduler.step(velocity,dt,action)
                result['action']=action
            return result
        finally:
            self.dit['runtime'].set_state(prior_rng)
            for module,mode in prior_modes.items(): module.training=mode

    def infer_world_model(self, sample, *, action_condition, video_seed=44):
        return self.infer_forcewm(sample, action_condition=action_condition,
                                  video_seed=video_seed, generate_action=False)

    def save_checkpoint(self,path,optimizer=None,step=None):
        payload={'schema':self.checkpoint_schema,'manifest':self.manifest,'model':self.state_dict(),'step':step}
        if optimizer is not None: payload['optimizer']=optimizer.state_dict()
        path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
        torch.save(payload,temp); temp.replace(path)

    def load_checkpoint(self,path,optimizer=None):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=self.checkpoint_schema or payload.get('manifest')!=self.manifest:
            raise ValueError('Checkpoint architecture/assets/protocol differ')
        self.load_state_dict(payload['model'],strict=True)
        if optimizer is not None: optimizer.load_state_dict(payload['optimizer'])
        return payload.get('step')
