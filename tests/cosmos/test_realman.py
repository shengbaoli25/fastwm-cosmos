import torch
from fastwam.cosmos.realman import RealManSource


def fixture():
    s=RealManSource.__new__(RealManSource)
    s.rows=[(ep,f,ep*40,40) for ep in range(2) for f in range(40)]
    s.prompts=['cup','cup']
    s.raw={k:torch.arange(80).float()[:,None].expand(-1,width).clone()
           for k,width in [('state',7),('action',7),('force',6)]}
    s.normalized={k:v.clone() for k,v in s.raw.items()}
    return s


def test_observed_history_and_shifted_targets_do_not_cross_episodes():
    s=fixture()
    for j in [0,10,31,39,40,41,71,79]:
        q=s[j]; ep,f,start,n=s.rows[j]
        for key, offsets in [('history_is_pad',range(-31,1)),('force_is_pad',range(1,33)),
                             ('action_is_pad',range(32)),('image_is_pad',range(0,33,4))]:
            assert q[key].tolist()==[not 0<=f+d<n for d in offsets]
        assert q['proprio'][0,:7].tolist()==[float(j)]*7
        assert q['history'][-1,0]==j
        valid=q['history'][~q['history_is_pad'],0]
        assert bool((valid<=j).all() & (valid>=start).all())
        valid=q['raw_wrench_future64'][~q['force_is_pad'],0]
        assert bool((valid>j).all() & (valid<start+n).all())
        assert not q['history'][q['history_is_pad']].any()
        assert not q['wrench_future'][q['force_is_pad']].any()
        assert not q['proprio'][:,-1].any()
        assert q['raw_wrench_future64'].dtype==torch.float64


def test_future_force_cannot_change_observed_history_or_action():
    s=fixture();before=s[8]
    s.normalized['force'][9:]+=1000
    after=s[8]
    for key in ['history','proprio','action']:
        assert torch.equal(before[key],after[key])
    assert not torch.equal(before['wrench_future'],after['wrench_future'])


def test_video_offsets_and_repeat_last_never_cross_episodes():
    s=fixture();s._video_episode=0
    s._frames=torch.arange(40).view(40,1,1,1).expand(-1,3,2,4)
    x=s.window(38)
    assert x.shape==(3,9,2,4)
    assert x[0,:,0,0].tolist()==[38]+[39]*8
