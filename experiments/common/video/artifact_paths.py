"""Resolve retained historical artifacts without rewriting their frozen bytes."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=4)
def _identity(repo):
    path = Path(repo) / 'data/metadata/experiment-archive/identity-map-20261110.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'artifact_aliases': {}}


def recorded_path(value, *, repo=ROOT):
    """Map a legacy recorded data path once; current code paths stay current."""
    repo = Path(repo).resolve()
    value = Path(value)
    # A POSIX-looking external path on Windows has a root but no drive. Preserve
    # that explicit external identity instead of attaching the repository drive.
    if value.root and not value.is_absolute():
        return value
    if value.is_absolute():
        try:
            name = value.relative_to(repo).as_posix()
        except ValueError:
            return value
    else:
        name = value.as_posix()
    for old, new in sorted(_identity(str(repo))['artifact_aliases'].items(), key=lambda p: -len(p[0])):
        if name == old or name.startswith(old + '/'):
            return repo / new / name[len(old):].lstrip('/')
    return repo / name


def frozen_path(repo, name, expected_sha256):
    """Require the original SHA, using an original source snapshot when needed."""
    repo = Path(repo).resolve()
    path = recorded_path(name, repo=repo)
    def matches(p):
        return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha256
    if matches(path):
        return path
    index = repo / 'data/metadata/experiment-archive/frozen-source-index-20261110.json'
    if index.exists():
        for rel in json.loads(index.read_text(encoding='utf-8')).get(expected_sha256, []):
            candidate = repo / rel
            if matches(candidate):
                return candidate
    raise ValueError('Original frozen bytes unavailable: ' + str(name) + ' SHA256=' + expected_sha256)
