"""Publish an explicitly mapped archive through Git objects, without an export tree.

The JSON plan names an existing publication repository and selected branches.
Only the publication refs/index are changed; source files and source Git are not.
Passing --push explicitly publishes the named refs, never all prepared branches.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def git(repository, args, data=None, env=None):
    result = subprocess.run(["git", "-C", str(repository)] + args,
                            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    repository = source_root / plan["publication_repository"]
    git_dir = Path(git(repository, ["rev-parse", "--absolute-git-dir"]).decode().strip())
    env = dict(os.environ, GIT_INDEX_FILE=str(git_dir / "archive-object-index"))
    commits, blobs = {}, []
    for spec in plan["branches"]:
        branch = spec["branch"]
        ref = "refs/heads/" + branch
        parent = git(repository, ["rev-parse", ref]).decode().strip()
        if parent != spec["expected_parent"]:
            raise RuntimeError("Publication branch moved: " + branch)
        replace_tree = bool(spec.get("replace_tree", False))
        git(repository, ["read-tree", "--empty"] if replace_tree else ["read-tree", parent], env=env)
        existing = git(repository, ["ls-tree", "-r", "-z", parent]).split(b"\0")
        paths = {}
        for record in existing:
            if not record:
                continue
            meta, path = record.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            paths[path.decode("utf-8")] = (mode, oid)
        changes = []
        for prefix in ([] if replace_tree else spec.get("remove_prefixes", [])):
            for path in paths:
                if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                    changes.append("0 " + "0" * 40 + "\t" + path + "\0")
        for item in spec["files"]:
            path = item["path"]
            if path.startswith("/") or ".." in Path(path).parts:
                raise ValueError("Invalid archive path: " + path)
            if "source" in item:
                data = (source_root / item["source"]).read_bytes()
            else:
                data = item["text"].encode("utf-8")
            if item.get("normalize_lf"):
                data = data.replace(b"\r\n", b"\n")
            oid = git(repository, ["hash-object", "-w", "--stdin"], data).decode().strip()
            mode = paths.get(path, ("100644", ""))[0]
            changes.append(mode + " " + oid + "\t" + path + "\0")
            blobs.append({"branch": branch, "path": path, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        git(repository, ["update-index", "-z", "--index-info"], "".join(changes).encode("utf-8"), env)
        tree = git(repository, ["write-tree"], env=env).decode().strip()
        if replace_tree:
            actual = set(git(repository, ["ls-tree", "-r", "--name-only", "-z", tree]).decode("utf-8").rstrip("\0").split("\0"))
            expected = {item["path"] for item in spec["files"]}
            if actual != expected:
                raise RuntimeError("Archive tree differs from the complete allowlist: " + branch)
        commit = git(repository, ["commit-tree", tree, "-p", parent, "-m", spec["message"]]).decode().strip()
        git(repository, ["update-ref", ref, commit, parent])
        commits[branch] = commit
        print(branch + " " + commit, flush=True)
    if args.push:
        refs = ["refs/heads/" + name + ":refs/heads/" + name for name in commits]
        git(repository, ["push", "origin"] + refs)
        remote = git(repository, ["ls-remote", "--heads", "origin"] + list(commits)).decode()
        for branch, oid in commits.items():
            if oid + "\trefs/heads/" + branch not in remote:
                raise RuntimeError("Remote verification failed: " + branch)
        print("Remote selected refs verified", flush=True)
    receipt = source_root / plan["receipt"]
    receipt.write_text(json.dumps({"commits": commits, "pushed": args.push,
                                   "publication_mode": "Git objects; no export directory created",
                                   "files": blobs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
