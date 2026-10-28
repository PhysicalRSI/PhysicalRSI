from physicalrsi.reporting import CoreArchive


def test_core_archive_is_plain_record_storage(tmp_path):
    archive = CoreArchive(tmp_path)
    archive.put("skill.json", {"id": "reach", "revision": "r1"})
    archive.put("skill.json", {"id": "reach", "revision": "r2"})
    assert archive.get("skill.json") == {"id": "reach", "revision": "r2"}
    assert archive.names() == ("skill.json",)
