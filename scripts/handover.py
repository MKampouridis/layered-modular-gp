"""Hand the project over between the two Macs through Box: one active machine at a time.

The same tool as llm-primitives-stgp (its decision D-055), adapted to this repository.

Code, docs and reports travel by git. What git does not carry (data, results and Claude's project
memory) travels through a hand-over folder in Box, `handover_dir` in config/paths.toml:

  .venv/bin/python scripts/handover.py status   # who holds the project; what is unsynced here
  .venv/bin/python scripts/handover.py push     # end of a session: git pushed, then copy to Box
  .venv/bin/python scripts/handover.py pull     # start of a session: git pull, then copy from Box

Rules that keep the two Macs from overwriting each other's work:
- `push` refuses unless git is clean and pushed, and unless this Mac holds the project: the hand-over
  in Box must be the one this Mac last pushed or pulled.
- `pull` refuses while this Mac has data, results or memory changed since its last hand-over (push
  them first), and verifies every copied file against the pushed checksums (Box may still be
  syncing). A pull then makes this Mac the holder, so the other Mac must pull before it can push.
- Nothing is deleted outright: a file removed on one Mac is moved to `_removed/<time>/` in the
  hand-over folder when the other side catches up.

The Box folder is only transport; each Mac works on its own local copy at the same path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE = REPO_ROOT / ".handover.json"  # this Mac's last hand-over (git-ignored)
MANIFEST = "HANDOVER.json"
SKIP = re.compile(r"(^|/)(\.DS_Store|__pycache__|\.tmp-[^/]*|[^/]*\.tmp)(/|$)")


class HandoverError(RuntimeError):
    pass


def memory_dir(repo: Path = REPO_ROOT) -> Path:
    """Claude Code's project memory for this repository (same path on both Macs)."""
    return (
        Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9-]", "-", str(repo)) / "memory"
    )


def roots(paths_file: Path | None = None) -> tuple[Path, dict[str, Path]]:
    """The hand-over folder and the local folders it carries."""
    source = Path(paths_file or os.environ.get("LMGP_PATHS") or REPO_ROOT / "config/paths.toml")
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    if not raw.get("handover_dir"):
        raise HandoverError(f"set handover_dir in {source} (a folder in Box)")

    def expand(value: str) -> Path:
        path = Path(os.path.expanduser(value))
        return path if path.is_absolute() else REPO_ROOT / path

    local = {
        "data": expand(raw.get("data_dir", "data")),
        "results": expand(raw.get("results_dir", "results")),
        "memory": memory_dir(),
    }
    return expand(raw["handover_dir"]), local


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(local: dict[str, Path], known: dict[str, dict] | None = None) -> dict[str, dict]:
    """{"<root>/<relative path>": {size, sha, mtime_ns}}; checksums reused when size and mtime match."""
    known = known or {}
    out = {}
    for name, root in local.items():
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
            rel = f"{name}/{path.relative_to(root).as_posix()}"
            if SKIP.search(rel):
                continue
            stat = path.stat()
            old = known.get(rel)
            same = old and old["size"] == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns
            sha = old["sha"] if same else _sha(path)
            out[rel] = {"size": stat.st_size, "sha": sha, "mtime_ns": stat.st_mtime_ns}
    return out


def _content(files: dict[str, dict]) -> dict[str, str]:
    return {rel: f["sha"] for rel, f in files.items()}


def _local_path(local: dict[str, Path], rel: str) -> Path:
    name, _, tail = rel.partition("/")
    return local[name] / tail


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".tmp-{path.name}")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git(*args: str) -> str:
    done = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    if done.returncode:
        raise HandoverError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def git_ready_to_push() -> str:
    if _git("status", "--porcelain"):
        raise HandoverError("git has uncommitted changes: commit (and push) them first")
    _git("fetch", "-q")
    if _git("rev-list", "@{u}..HEAD"):
        raise HandoverError("git has commits not pushed to GitHub: push them first")
    if _git("rev-list", "HEAD..@{u}"):
        raise HandoverError("GitHub has commits this Mac lacks: pull them first")
    return _git("rev-parse", "--short", "HEAD")


def git_pull() -> str:
    if _git("status", "--porcelain"):
        raise HandoverError("git has uncommitted changes here: commit or remove them first")
    _git("pull", "-q", "--ff-only")
    return _git("rev-parse", "--short", "HEAD")


def unsynced(local_files: dict[str, dict], state: dict | None) -> list[str]:
    """Files added, changed or removed here since this Mac's last hand-over."""
    before = _content(state["files"]) if state else {}
    now = _content(local_files)
    return sorted(rel for rel in before.keys() | now.keys() if before.get(rel) != now.get(rel))


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".tmp-{dst.name}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _set_aside(path: Path, box: Path, stamp: str, rel: str) -> None:
    target = box / "_removed" / stamp / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), target)


def push(box: Path, local: dict[str, Path], git: bool = True) -> dict:
    state = _read_json(STATE)
    current = _read_json(box / MANIFEST)
    if current and (not state or state.get("id") != current["id"]):
        raise HandoverError(
            f"{current['machine']} has handed over since this Mac last synced "
            f"({current['when_utc']}): pull first"
        )
    revision = git_ready_to_push() if git else ""
    files = scan(local, state["files"] if state else None)
    previous = _content(current["files"]) if current else {}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    copied = [rel for rel, f in files.items() if previous.get(rel) != f["sha"]]
    for rel in copied:
        _copy(_local_path(local, rel), box / "files" / rel)
    removed = sorted(previous.keys() - files.keys())
    for rel in removed:
        if (box / "files" / rel).exists():
            _set_aside(box / "files" / rel, box, stamp, rel)
    manifest = {
        "id": uuid.uuid4().hex,
        "machine": socket.gethostname(),
        "action": "push",
        "when_utc": stamp,
        "git": revision,
        "files": {rel: {"size": f["size"], "sha": f["sha"]} for rel, f in files.items()},
    }
    _write_json(box / MANIFEST, manifest)  # last, so a half-finished push is never pulled
    _write_json(STATE, {"id": manifest["id"], "files": files})
    return {"copied": len(copied), "removed": len(removed), "files": len(files)}


def pull(box: Path, local: dict[str, Path], git: bool = True, force: bool = False) -> dict:
    current = _read_json(box / MANIFEST)
    if not current:
        raise HandoverError(f"nothing has been handed over in {box} yet")
    state = _read_json(STATE)
    files = scan(local, state["files"] if state else None)
    target = current["files"]
    if state is None:  # first pull here: only files that differ from the hand-over count
        changed = sorted(r for r in files if target.get(r, {}).get("sha") != files[r]["sha"])
    else:
        changed = unsynced(files, state)
    if changed and not force:
        shown = "\n  ".join(changed[:10]) + ("\n  ..." if len(changed) > 10 else "")
        raise HandoverError(
            f"this Mac has {len(changed)} file(s) not handed over (push them first, or pass "
            f"--force to set them aside):\n  {shown}"
        )
    revision = git_pull() if git else ""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    fetched = [rel for rel, f in target.items() if files.get(rel, {}).get("sha") != f["sha"]]
    for rel in fetched:  # verify before touching anything local: Box may still be syncing
        src = box / "files" / rel
        if not src.exists() or _sha(src) != target[rel]["sha"]:
            raise HandoverError(f"{rel} in Box is missing or not yet synced: try again shortly")
    for rel in fetched:
        dst = _local_path(local, rel)
        if dst.exists():
            _set_aside(dst, box, stamp, f"{socket.gethostname()}/{rel}")
        _copy(box / "files" / rel, dst)
    extra = sorted(files.keys() - target.keys())
    for rel in extra:
        _set_aside(_local_path(local, rel), box, stamp, f"{socket.gethostname()}/{rel}")
    claim = {**current, "id": uuid.uuid4().hex, "machine": socket.gethostname(),
             "action": "pull", "when_utc": stamp}  # fmt: skip
    _write_json(box / MANIFEST, claim)  # this Mac now holds the project
    _write_json(STATE, {"id": claim["id"], "files": scan(local)})
    return {"fetched": len(fetched), "set_aside": len(extra), "git": revision,
            "from": current["machine"], "pushed_utc": current["when_utc"]}  # fmt: skip


def status(box: Path, local: dict[str, Path]) -> dict:
    current, state = _read_json(box / MANIFEST), _read_json(STATE)
    files = scan(local, state["files"] if state else None)
    return {
        "this_mac": socket.gethostname(),
        "last_handover": {k: current[k] for k in ("machine", "action", "when_utc", "git")}
        if current
        else None,
        "this_mac_holds_it": bool(current and state and state.get("id") == current["id"]),
        "unsynced_here": unsynced(files, state)[:20] if state else "never synced",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("action", choices=("status", "push", "pull"))
    parser.add_argument("--force", action="store_true",
                        help="pull: set this Mac's unsynced files aside in Box instead of refusing")  # fmt: skip
    args = parser.parse_args()
    try:
        box, local = roots()
        box.mkdir(parents=True, exist_ok=True)
        if args.action == "push":
            result = push(box, local)
        elif args.action == "pull":
            result = pull(box, local, force=args.force)
        else:
            result = status(box, local)
    except HandoverError as exc:
        sys.exit(f"handover {args.action}: {exc}")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
