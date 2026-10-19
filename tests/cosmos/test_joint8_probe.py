"""Exercise the real two-update probe with joint8 and the requested B64 geometry."""
from contextlib import nullcontext
from types import SimpleNamespace
import torch
from fastwam.cosmos.probe_updates import run_two_updates


def test_joint8_batch64_probe_updates_with_complete_short_schedule():
    class Model(torch.nn.Module):
        action_dim=8
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
        def training_loss(self,sample):
            loss=(self.weight*sample['action'].mean())**2
            return loss,{'action':float(loss.detach())}
    m=Model();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    acc=SimpleNamespace(accumulate=lambda _:nullcontext(),autocast=lambda:nullcontext(),
        backward=lambda x:x.backward(),sync_gradients=True,optimizer_step_was_skipped=False,
        gather=lambda x:x,clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    sample=dict(video_latents=torch.zeros(64,16,3,28,56),first_frame_latents=torch.zeros(64,16,1,28,56),
        action=torch.ones(64,32,8),proprio=torch.zeros(64,32,9),sample_index=torch.arange(64))
    t=SimpleNamespace(max_steps=1000,global_step=0,accelerator=acc,batch_size=64,model=m,
        train_loader=[sample,sample],batch_in_epoch=0,max_grad_norm=1.,optimizer=opt,
        scheduler=torch.optim.lr_scheduler.LinearLR(opt,total_iters=50))
    _,updates=run_two_updates(t,synchronize=lambda:None)
    assert t.global_step==2 and t.max_steps==1000 and t.scheduler.last_epoch==2
    assert all(u['batch_size']==64 and not u['optimizer_step_was_skipped'] for u in updates)
    assert opt.state[m.weight]['step']==2 and m.weight.item()<1
