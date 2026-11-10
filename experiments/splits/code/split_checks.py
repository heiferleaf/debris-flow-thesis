"""划分清单检查：schema、视频级无重叠、测试集不含生成帧。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path

ALLOWED_UNITS = {"video_id", "event_id"}
ALLOWED_TASKS = {"A", "B"}
ALLOWED_ORIGINS = {"real", "auto", "generated"}
REQUIRED_SPLITS = ("train", "val", "test")


def load_manifest(path: Path) -> dict[str, Any]:
    path = resolve_path(path, must_exist=True)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 根节点必须是 JSON 对象")
    return data


def _as_rows(split_name: str, rows: Any) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(rows, list):
        return [], [f"splits.{split_name} 必须是数组"]
    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"splits.{split_name}[{index}] 必须是对象")
            continue
        cleaned.append(row)
    return cleaned, errors


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """返回人类可读错误列表；空列表表示通过。"""
    errors: list[str] = []

    unit = data.get("unit")
    if unit not in ALLOWED_UNITS:
        errors.append("unit 必须是 video_id 或 event_id，禁止按 frame 划分")

    task = data.get("task")
    if task == "C":
        errors.append("task=C 禁止：不能把村庄静物照地点预报当作训练目标")
    elif task not in ALLOWED_TASKS:
        errors.append("task 必须是 A 或 B")

    if "name" not in data or not str(data.get("name", "")).strip():
        errors.append("缺少 name")

    splits = data.get("splits")
    if not isinstance(splits, dict):
        errors.append("缺少 splits 对象")
        return errors

    for split_name in REQUIRED_SPLITS:
        if split_name not in splits:
            errors.append(f"缺少 splits.{split_name}")

    ids_by_split: dict[str, set[str]] = {}
    origin_by_split: dict[str, list[str]] = defaultdict(list)

    for split_name, raw_rows in splits.items():
        rows, row_errors = _as_rows(split_name, raw_rows)
        errors.extend(row_errors)
        id_set: set[str] = set()
        for index, row in enumerate(rows):
            video_id = row.get("video_id")
            if not video_id or not str(video_id).strip():
                errors.append(f"splits.{split_name}[{index}] 缺少 video_id")
                continue
            video_id = str(video_id)
            if video_id in id_set:
                errors.append(f"splits.{split_name} 内部重复 video_id={video_id}")
            id_set.add(video_id)

            origin = row.get("label_origin")
            if origin not in ALLOWED_ORIGINS:
                errors.append(
                    f"splits.{split_name}[{index}] label_origin 必须是 real/auto/generated"
                )
            else:
                origin_by_split[split_name].append(str(origin))
        ids_by_split[split_name] = id_set

    ordered = [name for name in REQUIRED_SPLITS if name in ids_by_split]
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            overlap = sorted(ids_by_split[left] & ids_by_split[right])
            if overlap:
                errors.append(
                    f"划分泄漏：{left} 与 {right} 共享 video_id {overlap}"
                )

    generated_in_test = [
        "generated"
        for origin in origin_by_split.get("test", [])
        if origin == "generated"
    ]
    if generated_in_test:
        errors.append("测试集禁止 label_origin=generated（生成帧不得当测试标签）")

    return errors


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("用法: python -m experiments.splits.code.split_checks <manifest.json>", file=sys.stderr)
        return 2
    path = resolve_path(args[0])
    if not path.is_file():
        print(f"找不到文件: {path}", file=sys.stderr)
        return 2
    try:
        data = load_manifest(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"无法读取 {path}: {exc}", file=sys.stderr)
        return 2
    errors = validate_manifest(data)
    if errors:
        print(f"FAIL {path}")
        for item in errors:
            print(f"- {item}")
        return 1
    print(f"OK {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
