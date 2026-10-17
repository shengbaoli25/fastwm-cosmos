import json
import numpy as np
import pytest
from fastwam.cosmos.region_visualization import LATENT_GROUPS, overlay_frame, write_bundle, validate_trace


def fixture():
    frames=np.full((9,224,448,3),70,dtype=np.uint8)
    maps=np.zeros((2,2,32,2,14,14),dtype=np.float32)
    maps[:,:,:,0,3,2]=1;maps[:,:,:,1,10,11]=1
    return frames,maps


def test_camera_mapping_and_latent_group_hold():
    frames,maps=fixture();validate_trace(frames,maps)
    assert LATENT_GROUPS==[-1,0,0,0,0,1,1,1,1]
    first=overlay_frame(frames[0],maps,0)
    assert np.array_equal(first[48:],frames[0])
    actual=overlay_frame(frames[1],maps,1)
    # Peak at x=(2.5/14)*224=40, y=(3.5/14)*224=56 in BOTH views.
    assert tuple(actual[48+56,40])==(255,65,50)
    assert tuple(actual[48+56,224+40])==(255,65,50)
    maps[:,:,0,0,0,0]=float('nan')
    with pytest.raises(ValueError):validate_trace(frames,maps)


def test_video_bundle_contains_all_pairs_and_real_frame_count(tmp_path):
    import imageio.v2 as imageio
    frames,maps=fixture();write_bundle(tmp_path,frames,maps,{'scope':'synthetic-render-test-only'})
    assert np.load(tmp_path/'region_maps.npz')['attention'].shape==(2,2,32,2,14,14)
    with imageio.get_reader(tmp_path/'regions_overview.mp4') as reader:
        assert reader.count_frames()==9
        assert reader.get_data(0).shape==(272,448,3)
    html=(tmp_path/'viewer.html').read_text()
    assert '__DATA__' not in html and 'synthetic-render-test-only' in html
    assert json.loads((tmp_path/'metadata.json').read_text())['scope']=='synthetic-render-test-only'
