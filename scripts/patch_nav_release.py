#!/usr/bin/env python3
"""Stage a corrected local shard without changing an active release or uploading.

Replaces only the six navigation entries listed in a repair audit. Every other
member is streamed unchanged from the original shard. Emits a manifest fragment.
"""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--repair-report', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.archive.resolve() == args.output.resolve():
        ap.error('Use a separate output shard while release workers are active')
    report = json.loads(args.repair_report.read_text())
    if not report['applied']:
        raise ValueError('Repair must be applied and verified before patching a release')
    replacements, originals = {}, {}
    for item in report['files']:
        p = args.data_root / item['path']
        raw = p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['after_sha256']
        original = args.repair_report.parent / 'original' / item['path']
        original_raw = original.read_bytes()
        assert hashlib.sha256(original_raw).hexdigest() == item['before_sha256']
        name = item['path'] + '.gz'
        replacements[name] = gzip.compress(json.dumps(json.loads(raw), separators=(',', ':')).encode(), 6, mtime=0)
        originals[name] = json.loads(original_raw)
    before = args.archive.stat()
    found = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='uninavbench-nav-repair-') as temporary:
        staged = Path(temporary) / args.output.name
        with tarfile.open(args.archive, 'r:') as src, tarfile.open(staged, 'w') as dst:
            for member in src:
                info = copy.copy(member)
                if member.name in replacements:
                    assert json.loads(gzip.decompress(src.extractfile(member).read())) == originals[member.name]
                    payload = replacements[member.name]
                    info.size = len(payload)
                    dst.addfile(info, io.BytesIO(payload))
                    found.add(member.name)
                    print('Replaced', member.name, flush=True)
                else:
                    dst.addfile(info, src.extractfile(member) if member.isfile() else None)
        assert found == set(replacements), 'Missing repair entries in archive'
        assert (args.archive.stat().st_size, args.archive.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
        with tarfile.open(staged, 'r:') as verify:
            remaining = set(replacements)
            for member in verify:
                if member.name in remaining:
                    payload = verify.extractfile(member).read()
                    assert payload == replacements[member.name]
                    episodes = json.loads(gzip.decompress(payload))
                    assert len(episodes) == 2000 and all(len(e['instruction']) == 10 for e in episodes)
                    remaining.remove(member.name)
                    if not remaining:
                        break
            assert not remaining
        hasher = hashlib.sha256()
        with staged.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                hasher.update(block)
        checksum = hasher.hexdigest()
        print('Verified repaired entries; copying staged shard to', args.output, flush=True)
        partial = args.output.with_name(args.output.name + '.tmp')
        shutil.copyfile(staged, partial)
        partial.replace(args.output)
    fragment = {'file': args.output.name, 'bytes': args.output.stat().st_size,
                'sha256': checksum, 'replaces_archive': str(args.archive),
                'repaired_entries': sorted(found), 'published': False}
    args.output.with_suffix('.repair.json').write_text(json.dumps(fragment, indent=2) + '\n')
    print(json.dumps(fragment), flush=True)


if __name__ == '__main__':
    main()
