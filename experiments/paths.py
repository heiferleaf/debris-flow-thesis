"""Resolve recorded paths without rewriting historical experiment artifacts.

Only explicit directory aliases are relocated. Matching uses whole path components,
longest first, so a recorded run directory cannot be mistaken for source code.
Missing paths stay missing; no search by filename is performed.
"""
from __future__ import annotations

import os
import posixpath
from pathlib import Path, PureWindowsPath

REPO = Path(__file__).resolve().parents[1]
HISTORICAL_WINDOWS_REPO = "C:/Users/20443/Desktop/University_Learn/debris-flow-thesis"

# Longest matching boundary wins. Keep run aliases ahead of source aliases.
_ALIASES = {
    "experiments/repro/convnext/runs": "data/results/convnext",
    "experiments/repro/temporal/runs": "data/results/temporal",
    "experiments/repro/exp008/runs": "data/results/exp008",
    "experiments/repro/exp008/make_unit_folds.py": "experiments/splits/code/make_unit_folds.py",
    "experiments/repro/exp008/make_scene_folds.py": "experiments/splits/code/make_scene_folds.py",
    "experiments/tools/build_split_manifest.py": "experiments/splits/code/build_split_manifest.py",
    "experiments/tools/split_checks.py": "experiments/splits/code/split_checks.py",
    "experiments/tools/run_exp014.py": "experiments/runspecs/EXP-014/run.py",
    "experiments/repro/convnext": "experiments/common/convnext",
    "experiments/repro/temporal": "experiments/common/temporal",
    "experiments/repro/exp008": "experiments/common/video",
    ".claude/.temp": ".codex/docs/.temp",
    "data/frames-scenes": "data/final/video_windows/video2_segment",
    "data/frames-extneg": "data/process/video_frames/video2_segment",
    "data/frames": "data/process/video_frames/video1_segment",
}
# Only these seven frozen files moved; splits/code and its documentation remain.
_FILE_ALIASES = {
    f"experiments/splits/{name}": f"data/metadata/splits/{name}"
    for name in (
        "EXP-006-scene-folds.json",
        "EXP-006-scene-logo-folds.json",
        "EXP-006-session-folds.json",
        "EXP-008-units-folds-128.json",
        "EXP-008-units-folds.json",
        "EXP-008-units-iid-folds.json",
        "EXP-008-units-scene-folds.json",
    )
}
_CLOUD_ALIASES = {
    "/root/autodl-tmp/frames-v3": "data/process/EXP-008-units-v3-gap05/frames",
    "/root/frames": "data/process/EXP-008-units-v2/frames",
}


def _suffix(path: str, prefix: str, *, ignore_case: bool = False) -> str | None:
    p, key = (path.casefold(), prefix.casefold()) if ignore_case else (path, prefix)
    if p == key:
        return ""
    if p.startswith(key + "/"):
        return path[len(prefix) + 1:]
    return None


def resolve_path(value: str | os.PathLike, *, repo: Path | None = None,
                 must_exist: bool = False) -> Path:
    """Map a local, repository-relative, or explicitly recorded legacy path.

    Existing external absolute paths are returned as supplied. Repository aliases
    apply even when an old directory still exists, keeping writes in the new tree.
    Set must_exist for required read inputs to report the recorded and mapped paths.
    """
    raw = os.fspath(value)
    if not raw:
        raise ValueError("路径为空，无法确定输入文件")
    root = Path(repo) if repo is not None else REPO
    normalized = posixpath.normpath(raw.replace("\\", "/"))
    win_abs = PureWindowsPath(normalized).is_absolute()
    relative = None
    result = None
    for prefix, target in sorted(_CLOUD_ALIASES.items(), key=lambda kv: -len(kv[0])):
        tail = _suffix(normalized, prefix)
        if tail is not None:
            result = root / target / tail
            break
    if result is None:
        prefixes = [(root.as_posix().rstrip("/"), os.name == "nt"),
                    (HISTORICAL_WINDOWS_REPO, True), ("/root/repo", False)]
        for prefix, ignore_case in sorted(prefixes, key=lambda item: -len(item[0])):
            relative = _suffix(normalized, prefix, ignore_case=ignore_case)
            if relative is not None:
                break
        if relative is None and not (win_abs or normalized.startswith("/")):
            relative = normalized
        if relative is not None:
            for recorded, target in _FILE_ALIASES.items():
                if _suffix(relative, recorded, ignore_case=win_abs) == "":
                    relative = target
                    break
            else:
                for prefix, target in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
                    tail = _suffix(relative, prefix, ignore_case=win_abs)
                    if tail is not None:
                        relative = target + ("/" + tail if tail else "")
                        break
            result = root / relative
        else:
            result = Path(raw)
    if must_exist and not result.exists():
        raise FileNotFoundError(f"找不到路径：{raw}；当前对应位置：{result}")
    from experiments.common.video.artifact_paths import recorded_path
    return recorded_path(result, repo=root)
