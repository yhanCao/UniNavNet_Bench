#!/usr/bin/env python3
"""Finish an in-progress HF release, incorporating a staged nav repair safely.

Uploads completed shards while packaging continues. Installs and uploads the
repaired shard after packaging has finished, checks all remote LFS hashes, then
publishes an immutable manifest snapshot and dataset card in one commit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from huggingface_hub import HfApi, CommitOperationAdd


def atomic_json(path, data):
    tmp = path.with_name(path.name + '.upload.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def hash_file(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def remote_hash(info):
    lfs = info.lfs
    return lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)


def packaging_active(root):
    rows = subprocess.check_output(['ps', '-eo', 'args='], text=True).splitlines()
    return any('python' in row.split()[0] and 'scripts/build_release.py ' in row
               and str(root) in row for row in rows if row.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--repair-root', type=Path, required=True)
    ap.add_argument('--repo', default='yhncao/UniNavBench')
    args = ap.parse_args()
    root = args.root.resolve()
    source_scenes = {sp: {p.name for p in (args.data_root / sp).iterdir() if p.is_dir()}
                     for sp in ['train', 'val', 'test']}
    source_scenes['mini_test'] = {'0', '4', '6'}
    assert {k: len(v) for k, v in source_scenes.items()} == {'train': 500, 'val': 200, 'test': 100, 'mini_test': 3}
    repair_path = args.repair_root / 'release/train-005.tar'
    repair = json.loads((args.repair_root / 'release/train-005.repair.json').read_text())
    assert repair_path.stat().st_size == repair['bytes']
    assert hash_file(repair_path) == repair['sha256']
    api = HfApi()
    state_path = root / '.upload_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    # Check server metadata; never trust a local checkpoint alone.
    paths = [s['file'] for s in json.loads((root / 'manifest.json').read_text())['shards']]
    existing = {item.path: item for item in api.get_paths_info(args.repo, paths, repo_type='dataset')}
    manifest = json.loads((root / 'manifest.json').read_text())
    state = {s['file']: s['sha256'] for s in manifest['shards']
             if s['file'] in existing and existing[s['file']].size == s['bytes']
             and remote_hash(existing[s['file']]) == s['sha256']}
    atomic_json(state_path, state)
    print('[start]', args.repo, 'verified existing remote shards:', len(state), flush=True)

    def upload(shard):
        path = root / shard['file']
        if path.stat().st_size != shard['bytes']:
            raise ValueError('Local size changed: ' + shard['file'])
        child = ('from huggingface_hub import HfApi; import sys; '
                 'print(HfApi().upload_file(path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],'
                 'repo_id=sys.argv[3],repo_type="dataset",commit_message="Upload verified "+sys.argv[2]))')
        last_error = None
        for attempt in range(4):
            try:
                infos = api.get_paths_info(args.repo, [shard['file']], repo_type='dataset')
                if infos and infos[0].size == shard['bytes'] and remote_hash(infos[0]) == shard['sha256']:
                    print('[reuse]', shard['file'], flush=True)
                    return shard
                env = os.environ.copy()
                env['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
                if attempt:
                    env['HF_HUB_DISABLE_XET'] = '1'
                print('[upload]', shard['file'], 'attempt', attempt + 1, flush=True)
                run = subprocess.run([sys.executable, '-c', child, str(path), shard['file'], args.repo],
                                     env=env, capture_output=True, text=True, timeout=1800)
                if run.returncode:
                    raise RuntimeError(run.stderr[-1200:])
                info = api.get_paths_info(args.repo, [shard['file']], repo_type='dataset')[0]
                if info.size != shard['bytes'] or remote_hash(info) != shard['sha256']:
                    raise ValueError('Remote checksum mismatch: ' + shard['file'])
                print('[verified upload]', shard['file'], flush=True)
                return shard
            except Exception as exc:
                last_error = exc
                print('[retry]', shard['file'], type(exc).__name__, str(exc)[-400:], flush=True)
                time.sleep(5)
        raise RuntimeError('Upload exhausted retries: ' + shard['file']) from last_error

    pending = {}
    last_progress = time.monotonic()
    last_packaged = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            manifest = json.loads((root / 'manifest.json').read_text())
            shards = manifest['shards']
            groups = {sp: [scene for s in shards if s['split'] == sp for scene in s['scenes']]
                      for sp in source_scenes}
            for sp, ids in groups.items():
                assert len(ids) == len(set(ids)), 'Duplicate scene in manifest: ' + sp
                assert set(ids) <= source_scenes[sp], 'Unexpected scene in manifest: ' + sp
            packaged = {sp: len(ids) for sp, ids in groups.items()}
            if packaged != last_packaged:
                print('[packaged]', json.dumps(packaged), flush=True)
                last_packaged = packaged
                last_progress = time.monotonic()
            for name, future in list(pending.items()):
                if future.done():
                    shard = future.result()
                    state[name] = shard['sha256']
                    atomic_json(state_path, state)
                    del pending[name]
                    last_progress = time.monotonic()
            for shard in sorted(shards, key=lambda s: (s['split'] != 'mini_test', s['file'])):
                name = shard['file']
                if name == repair_path.name or name in pending or state.get(name) == shard['sha256']:
                    continue
                pending[name] = pool.submit(upload, deepcopy(shard))
            atomic_json(root / '.upload_progress.json', {
                'updated_utc': datetime.now(timezone.utc).isoformat(), 'phase': 'packaging_and_uploading',
                'packaged_scenes': packaged, 'expected_scenes': {sp: len(ids) for sp, ids in source_scenes.items()},
                'verified_remote_shards': len(state), 'pending_shards': sorted(pending),
                'repair_sha256': repair['sha256'],
            })
            complete = all(set(groups[sp]) == ids for sp, ids in source_scenes.items())
            if complete and not pending and not packaging_active(root):
                break
            if not packaging_active(root) and not complete:
                raise RuntimeError('Packaging stopped before all source scenes were included')
            if time.monotonic() - last_progress > 7200:
                raise TimeoutError('No packaging or upload progress for two hours')
            time.sleep(5)

    # The original packager owns manifest.json until it exits.
    manifest = json.loads((root / 'manifest.json').read_text())
    original_entry = next(s for s in manifest['shards'] if s['file'] == repair_path.name)
    if hash_file(root / repair_path.name) != repair['sha256']:
        print('[install repair]', repair_path.name, flush=True)
        temporary = root / (repair_path.name + '.repair.tmp')
        shutil.copyfile(repair_path, temporary)
        assert hash_file(temporary) == repair['sha256']
        backup = args.repair_root / 'original-release' / repair_path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(root / repair_path.name, backup)
        temporary.replace(root / repair_path.name)
    original_entry.update(bytes=repair['bytes'], sha256=repair['sha256'])
    manifest['version'] = '2026.11.14.1'
    manifest['repairs'] = {'train_220_229': 'repairs/train_220_229_20261114.json'}
    atomic_json(root / 'manifest.json', manifest)
    repaired = upload(original_entry)
    state[repaired['file']] = repaired['sha256']
    atomic_json(state_path, state)

    # Verify the complete release at one repository revision before publishing metadata.
    revision = api.dataset_info(args.repo).sha
    infos = {i.path: i for i in api.get_paths_info(args.repo, [s['file'] for s in manifest['shards']],
                                                 repo_type='dataset', revision=revision)}
    for shard in manifest['shards']:
        info = infos[shard['file']]
        assert info.size == shard['bytes'] and remote_hash(info) == shard['sha256'], shard['file']
    report = json.loads((args.repair_root / 'repair-report.json').read_text())
    for item in report['files']:
        item['cache_source'] = Path(item['cache_source']).name
    repairs = root / 'repairs/train_220_229_20261114.json'
    repairs.parent.mkdir(exist_ok=True)
    atomic_json(repairs, report)
    card = (root / 'README.md').read_text()
    note = ('\n\n## Completeness repair (2026-11-14)\n\n'
            'All train, val, test and mini_test shards are included in manifest version 2026.11.14.1. '
            'The six train/220 and train/229 navigation files each contain 2000 episodes of 10 subtasks. '
            'A generator shortfall was repaired by resampling 467 existing same-scene, same-level relation '
            'navigation tasks with full-day visibility evidence and conservative schedule checks. '
            'Original task prefixes and episode metadata are preserved; the 43 added tail episodes contain '
            'relation navigation only, restoring 4000 subtasks of each of the five navigation types per file. '
            'Per-task provenance is provided in [the repair report](repairs/train_220_229_20261114.json). '
            'These are replacement samples, not a recovery of original random selections.\n')
    if '## Completeness repair (2026-11-14)' not in card:
        card += note
    card_path = root / 'README.md'
    card_path.write_text(card)
    commit = api.create_commit(repo_id=args.repo, repo_type='dataset',
        commit_message='Publish complete UniNavBench release including repaired train scenes 220 and 229',
        parent_commit=revision, operations=[
            CommitOperationAdd(path_in_repo='manifest.json', path_or_fileobj=json.dumps(manifest, indent=2).encode()),
            CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card.encode()),
            CommitOperationAdd(path_in_repo='repairs/train_220_229_20261114.json', path_or_fileobj=str(repairs)),
        ])
    # Read back published metadata at this exact revision.
    import huggingface_hub
    published_path = huggingface_hub.hf_hub_download(repo_id=args.repo, repo_type='dataset',
                                                    filename='manifest.json', revision=commit.oid)
    assert json.loads(Path(published_path).read_text()) == manifest
    done = {'repo': args.repo, 'commit': commit.oid, 'commit_url': commit.commit_url,
            'shards': len(manifest['shards']), 'bytes': sum(s['bytes'] for s in manifest['shards']),
            'scenes': {sp: len(ids) for sp, ids in source_scenes.items()},
            'all_remote_hashes_verified': True, 'repair_sha256': repair['sha256']}
    atomic_json(root / 'upload_complete.json', done)
    atomic_json(root / '.upload_progress.json', dict(done, phase='complete'))
    print('[release published]', json.dumps(done), flush=True)


if __name__ == '__main__':
    main()
