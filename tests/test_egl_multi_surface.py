from PhysicalRSI_baselines.embodied_goodharts_law import robotwin_multi_surface as module


def test_multiple_visible_objects_deduplicate_by_geometry_not_size_rank(monkeypatch):
    monkeypatch.setattr(module, "public_world_camera", lambda observation, camera: {"images": {"rgb": "public-rgb"}})
    def geometry(source, marker):
        x, z = marker
        return {"status": "estimated_surface", "surface_quantiles": {
            "lower_05": [x - .02, -.02, z - .02], "upper_95": [x + .02, .02, z]},
            "full_object_extent_known": False}
    monkeypatch.setattr(module, "masked_surface_geometry", geometry)
    def segment(rgb, prompt, *, deadline):
        assert rgb == "public-rgb" and prompt == "block"
        return {"detections": [{"score": score, "mask": mask} for score, mask in
                [(.8, (0., .80)), (.95, (.001, .801)), (.9, (.15, .78)), (.85, (-.15, .76))]]}
    result = module.MultiSurfaceLocator(segmenter=segment).all({}, "block", deadline=10)
    assert len(result["objects"]) == 3
    assert result["objects"][0]["detection_score"] == .95
    assert all(x["frame"] == "world" and not x["semantic_identity_verified"] for x in result["objects"])


def test_no_detections_is_an_empty_estimate(monkeypatch):
    monkeypatch.setattr(module, "public_world_camera", lambda *args: {"images": {"rgb": None}})
    result = module.MultiSurfaceLocator(segmenter=lambda *a, **k: {"detections": []}).all({}, "block", deadline=10)
    assert result["objects"] == []
