import json

import numpy as np
import pytest

from fastwam.cosmos.query_visualization import overlay_query_frame, validate_query_trace, write_query_bundle


def fixture():
    frames = np.full((9, 224, 448, 3), 70, dtype=np.uint8)
    maps = np.zeros((2, 32, 2, 14, 14), dtype=np.float32)
    maps[:, :, 0, 3, 2] = .8
    maps[:, :, 1, 10, 11] = .2
    return frames, maps


def test_joint_camera_peaks_land_on_correct_pixels():
    frames, maps = fixture(); validate_query_trace(frames, maps)
    image = overlay_query_frame(frames, maps, 1)
    assert tuple(image[48 + 56, 40]) == (255, 90, 30)
    assert tuple(image[48 + 168, 224 + 184]) == (255, 90, 30)
    first = overlay_query_frame(frames, maps, 0)
    np.testing.assert_array_equal(first[48:272], frames[0])
    maps *= 2
    with pytest.raises(ValueError, match='joint-camera'): validate_query_trace(frames, maps)


def test_bundle_keeps_all_32_and_reference_label(tmp_path):
    import imageio.v2 as imageio
    frames, maps = fixture(); root = tmp_path / 'bundle'
    write_query_bundle(root, frames, maps, {'scope': 'synthetic-render-test-only'}, reference=frames)
    assert np.load(root / 'region_maps.npz')['attention'].shape == (2, 32, 2, 14, 14)
    with imageio.get_reader(root / 'query_00.mp4') as reader:
        assert reader.count_frames() == 9
        assert reader.get_data(0).shape == (520, 448, 3)
    html = (root / 'viewer.html').read_text()
    assert '__DATA__' not in html and 'GT latent' in html
    assert json.loads((root / 'metadata.json').read_text())['scope'] == 'synthetic-render-test-only'
    assert (root / 'all_queries_group1.png').exists() and (root / 'all_queries_group2.png').exists()
    with pytest.raises(FileExistsError): write_query_bundle(root, frames, maps, {})
