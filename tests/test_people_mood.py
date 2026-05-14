import numpy as np

from inference import people_mood as pm


def test_uniform_frame_has_no_face_crop():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    n, crop = pm.count_faces_and_largest_crop(rgb)
    assert n == 0
    assert crop is None


def test_analyze_people_returns_zero_faces_without_hf():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    import torch

    out = pm.analyze_people_mood(rgb, emotion_model_id=pm.DEFAULT_EMOTION_MODEL, torch_device=torch.device("cpu"))
    assert out["face_count"] == 0
    assert out["emotion_primary"] is None
