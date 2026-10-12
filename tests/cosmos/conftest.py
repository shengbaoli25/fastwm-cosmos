from pathlib import Path
import pytest
import torch
from fastwam.cosmos.native import native_imports,target_net_config
from fastwam.cosmos.model import CosmosForceWM

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='session',autouse=True)
def threads():
    torch.set_num_threads(2)


def tiny_net(depth=2):
    _,module,_,Net,_=native_imports(ROOT/'third_party/cosmos-predict2.5')
    cfg=target_net_config()
    cfg.update(max_img_h=8,max_img_w=8,max_frames=4,model_channels=32,num_blocks=depth,
        num_heads=2,adaln_lora_dim=8,crossattn_proj_in_channels=24,crossattn_emb_channels=16,
        use_wan_fp32_strategy=False,sac_config=module.SACConfig(mode='none'))
    return Net(**cfg)


def tiny_model(depth=2,checkpoint=True):
    torch.manual_seed(7)
    return CosmosForceWM(tiny_net(depth),manifest={'scope':'TEST-only','depth':depth},
        action_hidden=32,action_ffn=64,force_hidden=32,force_ffn=64,force_heads=2,
        memory_width=16,memory_heads=2,generated_video_steps=1,gradient_checkpointing=checkpoint)


def sample(batch=2):
    g=torch.Generator().manual_seed(98)
    normal=lambda *s:torch.randn(*s,generator=g)
    return dict(video_latents=normal(batch,16,3,4,4),first_frame_latents=normal(batch,16,1,4,4),
        cosmos_context=normal(batch,3,24),proprio=normal(batch,32,8),action=normal(batch,32,7),
        history=normal(batch,32,13),wrench_future=normal(batch,32,6),
        history_is_pad=torch.zeros(batch,32,dtype=torch.bool),history_channel_valid=torch.ones(batch,32,13,dtype=torch.bool),
        action_is_pad=torch.zeros(batch,32,dtype=torch.bool),force_is_pad=torch.zeros(batch,32,dtype=torch.bool),
        wrench_valid=torch.ones(batch,32,6,dtype=torch.bool),image_is_pad=torch.zeros(batch,9,dtype=torch.bool))
