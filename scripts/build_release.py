"""Pack the dataset into the shards that `uninavbench download` fetches.

    python scripts/build_release.py DATA_ROOT OUT_DIR --resume [--workers 2]

Only what evaluation needs goes into a shard: episodes/<level>/{nav,eqa}.json.gz (compact JSON),
dynamic_strategy/strategy_final_{small,all}.json and the goal images referenced by image-goal navigation.
OUT_DIR then holds <split>-NNN.tar and manifest.json, ready to upload as a dataset repository:

mini_test contains test scenes 0, 4 and 6 by default, preserving their original test/ paths.

    huggingface-cli upload <user>/UniNavBench OUT_DIR . --repo-type dataset
    modelscope upload <user>/UniNavBench OUT_DIR --repo-type dataset
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import time
import shutil
import tempfile
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock

LEVELS = ("static", "small", "all")
_IO_LOCK = None


def initialize_worker(io_lock):
    global _IO_LOCK
    _IO_LOCK = io_lock


def read_bytes(path):
    # Serialize source reads to avoid competing seeks on external disks.
    with _IO_LOCK if _IO_LOCK is not None else nullcontext():
        with open(path, "rb") as f:
            return f.read()


def add_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0  # reproducible archives
    tar.addfile(info, io.BytesIO(data))


def load(path):
    if os.path.exists(path + ".gz"):
        return json.loads(gzip.decompress(read_bytes(path + ".gz")))
    return json.loads(read_bytes(path))


def add_scene(tar, root, split, scene):
    base = os.path.join(root, split, scene)
    images = set()
    for level in LEVELS:
        for task in ("nav", "eqa"):
            episodes = load(os.path.join(base, "episodes", level, f"{task}.json"))
            if task == "nav":
                images.update(n for ep in episodes for n in ep["image_name"] if n)
            data = gzip.compress(json.dumps(episodes, separators=(",", ":")).encode(), 6, mtime=0)
            add_bytes(tar, f"{split}/{scene}/episodes/{level}/{task}.json.gz", data)
    for level in ("small", "all"):
        strategy = load(os.path.join(base, "dynamic_strategy", f"strategy_final_{level}.json"))
        add_bytes(tar, f"{split}/{scene}/dynamic_strategy/strategy_final_{level}.json",
                  json.dumps(strategy, separators=(",", ":")).encode())
    image_dir = os.path.join(root, "attributes", split, scene, "images")
    for name in sorted(images):
        path = os.path.join(image_dir, name)
        if not os.path.exists(path):
            path = os.path.join(image_dir, name.replace("|", "_"))
        add_bytes(tar, f"attributes/{split}/{scene}/images/{name}", read_bytes(path))
    return len(images)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_shard(root, out, split, group, name, staging_dir=None):
    source_split = "test" if split == "mini_test" else split
    path = os.path.join(out, name)
    with tempfile.TemporaryDirectory(prefix="uninavbench-", dir=staging_dir or out) as temporary:
        staged = os.path.join(temporary, name)
        with tarfile.open(staged, "w") as tar:
            n_images = 0
            for scene in group:
                n_images += add_scene(tar, root, source_split, scene)
                print(f"[pack] {name}: scene {scene} complete", flush=True)
        checksum = sha256(staged)
        with _IO_LOCK if _IO_LOCK is not None else nullcontext():
            shutil.move(staged, path + ".tmp")
            os.replace(path + ".tmp", path)
    shard = {"file": name, "split": split, "scenes": group, "bytes": os.path.getsize(path),
             "sha256": checksum, "images": n_images}
    if split == "mini_test":
        shard["source_split"] = "test"
    return shard


def save_manifest(path, manifest):
    manifest["shards"].sort(key=lambda s: s["file"])
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--splits", nargs="+", default=["mini_test", "train", "val", "test"],
                    choices=["train", "val", "test", "mini_test"])
    ap.add_argument("--scenes-per-shard", type=int, default=25)
    ap.add_argument("--mini-test-scenes", nargs=3, default=["0", "4", "6"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--staging-dir", default=None, help="Fast temporary disk for building shards.")
    ap.add_argument("--resume", action="store_true", help="Reuse existing shards after SHA-256 verification.")
    ap.add_argument("--version", default=time.strftime("%Y.%m.%d"))
    args = ap.parse_args()
    if args.scenes_per_shard < 1 or args.workers < 1:
        ap.error("--scenes-per-shard and --workers must be positive")
    if len(set(args.mini_test_scenes)) != 3:
        ap.error("--mini-test-scenes must contain three distinct scene ids")
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"shards": []}
    manifest["version"] = args.version
    manifest["layout"] = "{split}/{scene}/episodes/{level}/{nav,eqa}.json.gz, {split}/{scene}/dynamic_strategy/" \
                         "strategy_final_{small,all}.json, attributes/{split}/{scene}/images/*.png"
    previous = {s["file"]: s for s in manifest["shards"]}
    manifest["shards"] = [s for s in manifest["shards"] if s["split"] not in args.splits]
    jobs = []
    for split in args.splits:
        source_split = "test" if split == "mini_test" else split
        scenes = sorted(args.mini_test_scenes if split == "mini_test" else
                        [n for n in os.listdir(os.path.join(args.root, split))
                         if os.path.isdir(os.path.join(args.root, split, n))], key=int)
        for scene in scenes:
            base = os.path.join(args.root, source_split, scene)
            for level in LEVELS:
                for task in ("nav", "eqa"):
                    path = os.path.join(base, "episodes", level, task + ".json")
                    if not (os.path.isfile(path) or os.path.isfile(path + ".gz")):
                        raise FileNotFoundError(path)
            for level in ("small", "all"):
                path = os.path.join(base, "dynamic_strategy", f"strategy_final_{level}.json")
                if not (os.path.isfile(path) or os.path.isfile(path + ".gz")):
                    raise FileNotFoundError(path)
        size = 3 if split == "mini_test" else args.scenes_per_shard
        for k in range(0, len(scenes), size):
            group = scenes[k:k + size]
            name = f"{split}-{k // size:03d}.tar"
            path = os.path.join(args.out, name)
            old = previous.get(name)
            if (args.resume and old and old["scenes"] == group and os.path.isfile(path)
                    and old.get("source_split", old["split"]) == source_split
                    and os.path.getsize(path) == old["bytes"] and sha256(path) == old["sha256"]):
                manifest["shards"].append(old)
                print(f"[reuse] {name}", flush=True)
            else:
                jobs.append((args.root, args.out, split, group, name, args.staging_dir))
    save_manifest(manifest_path, manifest)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(Lock(),)) as pool:
        futures = [pool.submit(build_shard, *job) for job in jobs]
        for future in as_completed(futures):
            shard = future.result()
            manifest["shards"].append(shard)
            save_manifest(manifest_path, manifest)
            print(json.dumps(shard), flush=True)
    total = sum(s["bytes"] for s in manifest["shards"])
    print(f"{len(manifest['shards'])} shards, {total / 1e9:.1f} GB -> {args.out}")


if __name__ == "__main__":
    main()
