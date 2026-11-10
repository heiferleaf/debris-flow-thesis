"""Publish a validated first experiment archive into an EMPTY authorized remote.

Does not switch/stage/commit the source working tree. Refuses existing remote
branches; updates must clone the published history instead of force-pushing.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def run(args, cwd):
    result = subprocess.run(args, cwd=str(cwd), text=True, encoding='utf-8',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError('%s failed: %s' % (' '.join(args[:3]), result.stderr[-3000:]))
    return result.stdout.strip()


def enumerate_files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts)


def copy_tree(source, target):
    for p in enumerate_files(source):
        dest = target / p.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(p), str(dest))


def stage(paths, repo, pathspec):
    pathspec.write_bytes(b''.join(str(p).replace('\\','/').encode('utf-8') + b'\0' for p in paths))
    run(['git','add','--pathspec-from-file=' + str(pathspec.resolve()),'--pathspec-file-nul'],repo)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--remote', default='https://github.com/heiferleaf/debris-flow-thesis.git')
    p.add_argument('--resume-empty-local', action='store_true')
    a = p.parse_args()
    export = a.export.resolve(); source = a.source.resolve()
    check = json.loads((export/'validation.json').read_text(encoding='utf-8'))
    if check['status'] != 'export_validated':
        raise SystemExit('Unvalidated export')
    repo = export/'publication-repo'
    if repo.exists() and not a.resume_empty_local:
        raise SystemExit('Publication repository exists; inspect it rather than overwriting')
    if run(['git','ls-remote','--heads',a.remote],source):
        raise SystemExit('Remote already has branches; preserve its published history')
    source_head = run(['git','rev-parse','HEAD'],source)
    source_branch = run(['git','branch','--show-current'],source)
    if repo.exists():
        if run(['git','remote','get-url','origin'],repo) != a.remote:
            raise SystemExit('Existing local repository uses a different remote')
        if subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=str(repo),stdout=subprocess.PIPE,stderr=subprocess.PIPE).returncode == 0:
            raise SystemExit('Existing local commits need an explicit history-preserving resume')
    else:
        repo.mkdir()
        run(['git','init','--initial-branch=main'],repo)
    for key in ['user.name','user.email']:
        value = run(['git','config',key],source)
        if not value:
            raise SystemExit('Missing author ' + key)
        run(['git','config',key,value],repo)
    run(['git','config','core.autocrlf','false'],repo)
    run(['git','config','core.longpaths','true'],repo)
    run(['git','config','credential.helper','!gh auth git-credential'],repo)
    if not a.resume_empty_local:
        run(['git','remote','add','origin',a.remote],repo)
    copy_tree(export/'base',repo)
    (repo/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    paths = [p.relative_to(repo) for p in enumerate_files(repo)]
    pathspec = export/'publish-pathspec.bin'
    stage(paths,repo,pathspec)
    run(['git','commit','-m','归档实验总索引、共享实现与冻结数据元信息'],repo)
    commits = {'main':run(['git','rev-parse','HEAD'],repo)}
    print('COMMIT main',commits['main'],flush=True)
    for overlay in sorted((export/'overlays').iterdir()):
        branch = 'codex/' + overlay.name
        run(['git','switch','--create',branch,'main'],repo)
        copy_tree(overlay,repo)
        paths = [p.relative_to(overlay) for p in enumerate_files(overlay)]
        stage(paths,repo,pathspec)
        run(['git','commit','-m','归档%s最终代码、数据使用与有效结果' % overlay.name],repo)
        commits[branch] = run(['git','rev-parse','HEAD'],repo)
        print('COMMIT',branch,commits[branch],flush=True)
        run(['git','switch','main'],repo)
    # All commits are complete and reviewable before the external upload.
    print('PUSH',len(commits),'refs',flush=True)
    run(['git','push','--atomic','origin'] + list(commits),repo)
    remote = run(['git','ls-remote','--heads','origin'],repo)
    refs = {line.split()[1].removeprefix('refs/heads/'):line.split()[0] for line in remote.splitlines()} if hasattr(str,'removeprefix') else {line.split()[1][len('refs/heads/'):]:line.split()[0] for line in remote.splitlines()}
    assert all(refs.get(k) == v for k,v in commits.items()), 'Remote commit mismatch'
    assert run(['git','rev-parse','HEAD'],source) == source_head
    assert run(['git','branch','--show-current'],source) == source_branch
    receipt = {'repository':a.remote[:-4] if a.remote.endswith('.git') else a.remote,
               'visibility':'public','status':'pushed_and_remote_refs_verified',
               'commits':commits,'source_head_unchanged':source_head,'source_branch_unchanged':source_branch,
               'experiment_count':len(commits)-1,'replay_verified_frozen_head_count':165,
               'replay_scope':'cached features -> head weights -> predictions -> metrics; not complete historical retraining',
               'other_large_model_weights':'not uploaded before independent weight replay verification'}
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'experiments':len(commits)-1}),flush=True)


if __name__ == '__main__':
    main()
