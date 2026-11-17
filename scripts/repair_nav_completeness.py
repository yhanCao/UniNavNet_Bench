#!/usr/bin/env python3
"""Fill the relation-navigation shortfall in train/220 and train/229.

Uses existing same-scene/level relation tasks certified by visibility timelines.
Existing records and task prefixes are preserved. No simulator or API is called.
Default is review-only; --apply backs up original bytes before atomic replacement.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import zlib

FIELDS = ('instruction', 'image_name', 'id', 'task_type', 'object_type')
RELATION = re.compile(r'^Find the (\w+) which is (.+?) the (\w+)$')
ROOM = re.compile(r'^Find the (\w+) in the (\w+)$')
EXCLUDED = {'Wall', 'Floor', 'Doorway', 'Window', 'Doorframe'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def changed_objects(schedule):
    changed = set()
    for group in schedule.values():
        for event in group:
            if not event.get('success', True):
                continue
            # Conservatively exclude targets/references affected by any scheduled action.
            if event.get('objectId'):
                changed.add(event['objectId'])
            changed.update(item['objectId'] for item in event.get('resolved') or [])
    return changed


def relation_is_safe(ep, i, cache, changed):
    if ep['task_type'][i] != 2 or ep['image_name'][i] is not None:
        return False
    oid, text = ep['id'][i], ep['instruction'][i]
    timeline = cache['timeline'].get(oid)
    if not oid or oid.split('|')[0] in EXCLUDED or oid in changed:
        return False
    if not timeline or not all(value for _, value in timeline):
        return False
    expected_type = cache['types'].get(oid)
    room, relation = ROOM.fullmatch(text), RELATION.fullmatch(text)
    match = room or relation
    if not match or match.group(1) != expected_type:
        return False
    if relation:
        ref_type = relation.group(3)
        if ref_type == expected_type or ref_type in {o.split('|')[0] for o in changed}:
            return False
    return True


def repair(episodes, cache, schedule, scene, level, expected_episodes=2000):
    target_per_type = expected_episodes * 2
    for e in episodes:
        n = len(e['instruction'])
        if n > 10 or n < 1 or any(len(e[f]) != n for f in FIELDS):
            raise ValueError('Inconsistent task arrays')
        if e['scene_type'] != 'train' or e['scene_id'] != scene or e['level'] != level:
            raise ValueError('Wrong scene or level')
    if not episodes or len(episodes) > expected_episodes or any(len(e['instruction']) != 10 for e in episodes[:-1]):
        raise ValueError('Only a short final episode and missing tail episodes are supported')
    counts = Counter(t for e in episodes for t in e['task_type'])
    if set(counts) != {1, 2, 3, 4, 5} or any(counts[t] != target_per_type for t in [1, 3, 4, 5]):
        raise ValueError('Expected a shortfall confined to relation navigation')
    needed = target_per_type - counts[2]
    if needed < 0:
        raise ValueError('Too many relation tasks')
    changed = changed_objects(schedule)
    unique = {}
    for k, ep in enumerate(episodes):
        for i in range(len(ep['instruction'])):
            if relation_is_safe(ep, i, cache, changed):
                unique.setdefault((ep['id'][i], ep['instruction'][i]), (k, i))
    pool = list(unique.values())
    if needed and not pool:
        raise ValueError('No certified relation tasks available')
    rng = random.Random(zlib.crc32(f'train|{scene}|{level}|relation-completeness-v1'.encode()))
    result, additions, templates = deepcopy(episodes), [], []
    last_time = max([int(e['time']) for group in schedule.values() for e in group] or [360])
    for _ in range(needed):
        if len(result[-1]['instruction']) == 10:
            tk = rng.randrange(len(episodes))
            ep = deepcopy(episodes[tk])
            for f in FIELDS:
                ep[f] = []
            if level != 'static':
                index = len(result)
                ep['time'] = random.Random(zlib.crc32(f'train|{scene}|{level}|{index}|start'.encode())).randint(360, max(360, last_time - 120))
            result.append(ep)
            templates.append({'episode': len(result) - 1, 'pose_source_episode': tk})
        destination = result[-1]
        available = [(k, i) for k, i in pool if episodes[k]['instruction'][i] not in destination['instruction']]
        sk, si = rng.choice(available or pool)
        for f in FIELDS:
            destination[f].append(deepcopy(episodes[sk][f][si]))
        additions.append({'episode': len(result) - 1, 'subtask': len(destination['instruction']) - 1,
                          'source_episode': sk, 'source_subtask': si})
    assert len(result) == expected_episodes
    assert all(len(ep[f]) == 10 for ep in result for f in FIELDS)
    assert Counter(t for ep in result for t in ep['task_type']) == Counter({t: target_per_type for t in range(1, 6)})
    # Existing task arrays are untouched prefixes; every other original field is identical.
    for old, new in zip(episodes, result):
        for key, value in old.items():
            assert (new[key][:len(value)] == value) if key in FIELDS else (new[key] == value)
    for item in additions:
        assert relation_is_safe(result[item['episode']], item['subtask'], cache, changed)
    sequences = [tuple(e['instruction']) for e in result]
    for i in range(len(episodes), len(result)):
        assert sequences[i] not in sequences[:i], 'Duplicate full episode generated'
    return result, {'scene': scene, 'level': level, 'episodes_before': len(episodes),
                    'episodes_after': len(result), 'added_subtasks': needed,
                    'unique_safe_relation_pool': len(pool), 'task_counts_before': dict(counts),
                    'task_counts_after': {t: target_per_type for t in range(1, 6)},
                    'new_episode_pose_templates': templates, 'additions': additions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--backup-dir', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    prepared, report = [], {'method': 'same-scene-level certified relation resampling', 'files': []}
    for scene in [220, 229]:
        for level in ['static', 'small', 'all']:
            rel = Path('train') / str(scene) / 'episodes' / level / 'nav.json'
            p = args.root / rel
            raw = p.read_bytes()
            cache_path = args.cache_dir / f'train_{scene}_{level}.json'
            cache_raw = cache_path.read_bytes()
            strategy = args.root / 'train' / str(scene) / 'dynamic_strategy' / f'strategy_final_{level}.json'
            schedule_raw = strategy.read_bytes() if level != 'static' else b'{}'
            result, item = repair(json.loads(raw), json.loads(cache_raw), json.loads(schedule_raw), scene, level)
            data = (json.dumps(result, ensure_ascii=False, indent=4) + '\n').encode()
            item.update(path=str(rel), before_sha256=sha256(raw), after_sha256=sha256(data),
                        cache_sha256=sha256(cache_raw), cache_source=str(cache_path),
                        schedule_sha256=sha256(schedule_raw))
            report['files'].append(item)
            if item["added_subtasks"]:
                prepared.append((p, rel, raw, data))
            print(json.dumps({k: v for k, v in item.items() if k not in ['additions', 'new_episode_pose_templates']}, ensure_ascii=False), flush=True)
    report['applied'] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Write the full audit trail before any data mutation.
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not args.apply:
        return
    for p, rel, raw, data in prepared:
        if p.read_bytes() != raw:
            raise RuntimeError(f'Concurrent modification: {p}')
        backup = args.backup_dir / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists() and backup.read_bytes() != raw:
            raise RuntimeError(f'Existing backup differs: {backup}')
        if not backup.exists():
            shutil.copy2(p, backup)
        if p.with_suffix('.json.gz').exists():
            raise RuntimeError('Unexpected gzip source; repair both encodings explicitly before applying')
    for p, rel, raw, data in prepared:
        tmp = p.with_name(p.name + '.repair.tmp')
        tmp.write_bytes(data)
        tmp.replace(p)
    report['applied'] = True
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Applied {len(prepared)} files; original bytes and detailed provenance retained.', flush=True)


if __name__ == '__main__':
    main()
