"""The Box hand-over between the two Macs (scripts/handover.py)."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("handover", REPO_ROOT / "scripts" / "handover.py")
handover = importlib.util.module_from_spec(spec)
sys.modules["handover"] = handover
spec.loader.exec_module(handover)
HandoverError = handover.HandoverError


class Mac:
    """One Mac: its own data, results, memory and hand-over state; Box is shared."""

    def __init__(self, root, name, box, monkeypatch):
        self.name, self.box, self.mp = name, box, monkeypatch
        self.local = {k: root / name / k for k in ("data", "results", "memory")}
        self.state = root / name / ".handover.json"

    def _as_me(self):
        self.mp.setattr(handover, "STATE", self.state)
        self.mp.setattr(handover.socket, "gethostname", lambda: self.name)

    def push(self):
        self._as_me()
        return handover.push(self.box, self.local, git=False)

    def pull(self, force=False):
        self._as_me()
        return handover.pull(self.box, self.local, git=False, force=force)

    def write(self, rel, text):
        path = handover._local_path(self.local, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def read(self, rel):
        path = handover._local_path(self.local, rel)
        return path.read_text() if path.exists() else None


@pytest.fixture
def macs(tmp_path, monkeypatch):
    box = tmp_path / "box"
    box.mkdir()
    return Mac(tmp_path, "air", box, monkeypatch), Mac(tmp_path, "pro", box, monkeypatch)


def test_work_moves_between_macs_and_back(macs):
    air, pro = macs
    air.write("data/tierA/SPX.parquet", "prices")
    air.write("results/campaigns/c/tasks.json", "plan")
    air.write("memory/MEMORY.md", "notes")
    assert air.push()["copied"] == 3
    assert pro.pull()["fetched"] == 3
    assert pro.read("results/campaigns/c/tasks.json") == "plan"
    pro.write("results/campaigns/c/r1.json", "record")
    pro.write("memory/MEMORY.md", "notes, updated")
    assert pro.push()["copied"] == 2
    air.pull()
    assert air.read("results/campaigns/c/r1.json") == "record"
    assert air.read("memory/MEMORY.md") == "notes, updated"


def test_the_mac_that_did_not_pull_cannot_push(macs):
    air, pro = macs
    air.write("results/a.json", "1")
    air.push()
    pro.pull()  # pro now holds the project
    air.write("results/b.json", "2")
    with pytest.raises(HandoverError, match="pull first"):
        air.push()


def test_pull_refuses_while_this_mac_has_work_not_handed_over(macs):
    air, pro = macs
    air.write("results/a.json", "1")
    air.push()
    pro.pull()
    pro.write("results/a.json", "changed on pro")
    pro.push()
    air.write("results/new.json", "made on air after its push")
    with pytest.raises(HandoverError, match="not handed over"):
        air.pull()
    assert air.read("results/a.json") == "1"  # untouched
    air.pull(force=True)  # sets the unsynced file aside in Box rather than losing it
    assert air.read("results/a.json") == "changed on pro"
    assert air.read("results/new.json") is None
    kept = list((air.box / "_removed").rglob("new.json"))
    assert len(kept) == 1
    assert kept[0].read_text() == "made on air after its push"
    overwritten = [f.read_text() for f in (air.box / "_removed").rglob("a.json")]
    assert overwritten == ["1"]  # air's copy, replaced by pro's, is kept too


def test_first_pull_refuses_to_overwrite_different_local_files(macs):
    air, pro = macs
    air.write("results/a.json", "air's")
    air.push()
    pro.write("results/a.json", "pro's own")
    with pytest.raises(HandoverError, match="not handed over"):
        pro.pull()
    assert pro.read("results/a.json") == "pro's own"


def test_a_file_still_syncing_in_box_stops_the_pull_before_anything_changes(macs):
    air, pro = macs
    air.write("results/a.json", "1")
    air.write("results/b.json", "2")
    air.push()
    (air.box / "files" / "results" / "b.json").write_text("half-synced")
    with pytest.raises(HandoverError, match="not yet synced"):
        pro.pull()
    assert pro.read("results/a.json") is None


def test_a_removal_reaches_the_other_mac_and_the_file_is_kept_in_box(macs):
    air, pro = macs
    air.write("results/old/r.json", "old")
    air.push()
    pro.pull()
    handover._local_path(pro.local, "results/old/r.json").unlink()
    assert pro.push()["removed"] == 1
    air.pull()
    assert air.read("results/old/r.json") is None
    assert [p.read_text() for p in (air.box / "_removed").rglob("r.json")] == ["old", "old"]


def test_pushing_records_the_mac_and_never_copies_junk(macs):
    air, _ = macs
    air.write("results/a.json", "1")
    air.write("results/.DS_Store", "x")
    air.write("results/__pycache__/m.pyc", "x")
    air.push()
    manifest = json.loads((air.box / "HANDOVER.json").read_text())
    assert manifest["machine"] == "air"
    assert list(manifest["files"]) == ["results/a.json"]


def test_memory_folder_matches_claude_codes_project_naming():
    folder = handover.memory_dir(Path("/Users/mkampo/code/layered-modular-gp"))
    assert folder.parts[-3:] == ("projects", "-Users-mkampo-code-layered-modular-gp", "memory")
