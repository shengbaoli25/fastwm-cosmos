from pathlib import Path
import ast, copy
import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as hf_network
from fastwam.cosmos.reason1_native import native_network,Reason1SDPA,full_concat

ROOT=Path(__file__).resolve().parents[2]


def config():
    return Qwen2_5_VLConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,vocab_size=100,use_sliding_window=False,
        rope_scaling={'type':'mrope','mrope_section':[2,3,3]},
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
            out_hidden_size=32,patch_size=14,temporal_patch_size=2,spatial_merge_size=2,
            fullatt_block_indexes=[0]))


def test_native_text_states_match_independent_hf_implementation():
    torch.manual_seed(6)
    c=config();native=Reason1SDPA(native_network(ROOT/'third_party/cosmos-predict2.5'),c).eval()
    text_type=getattr(hf_network,'Qwen2_5_VLTextModel',hf_network.Qwen2_5_VLModel)
    hf=text_type(getattr(c,'text_config',c)).eval()
    hf.load_state_dict(native.model.state_dict(),strict=True)
    ids=torch.tensor([[1,2,5,0,0,0],[7,4,2,6,0,0]])
    with torch.no_grad():
        a=native.model(ids,output_hidden_states=True,use_cache=False).hidden_states
        mask=torch.full((2,1,6,6),float('-inf')).triu(1)
        b=hf(ids,attention_mask=mask,output_hidden_states=True,use_cache=False).hidden_states
    assert len(a)==len(b)==3
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(full_concat(a),full_concat(b),rtol=3e-5,atol=3e-6)


def test_full_concat_matches_pinned_text_encoder_formula():
    tree=ast.parse((ROOT/'third_party/cosmos-predict2.5/cosmos_predict2/_src/predict2/text_encoders/text_encoder.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TextEncoder')
    f=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='mean_normalize'))
    f.decorator_list=[]; env={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'upstream_normalize','exec'),env)
    x=[torch.randn(2,8,32,dtype=torch.bfloat16) for _ in range(29)]
    expected=torch.cat([env['mean_normalize'](q) for q in x[1:]],dim=-1)
    assert torch.equal(expected,full_concat(x))


def test_strict_state_and_meta_buffers_materialize():
    net=native_network(ROOT/'third_party/cosmos-predict2.5');reference=Reason1SDPA(net,config())
    with torch.device('meta'):m=Reason1SDPA(net,config())
    m.load_state_dict(reference.state_dict(),strict=True,assign=True)
    m.materialize_rotary(net)
    assert not any(x.is_meta for x in list(m.parameters())+list(m.buffers()))
