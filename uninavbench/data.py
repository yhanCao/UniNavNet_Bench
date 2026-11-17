"""Download and locate the UniNavBench dataset.

The dataset is published as tar shards of 25 scenes plus ``manifest.json`` on the Hugging Face Hub
(default) or ModelScope. ``download`` fetches the shards of the requested splits / scenes, checks their
sha256 and extracts them into the data directory:

    <data_dir>/{train,val,test}/<scene>/episodes/{static,small,all}/{nav,eqa}.json.gz
    <data_dir>/{train,val,test}/<scene>/dynamic_strategy/strategy_final_{small,all}.json
    <data_dir>/attributes/{train,val,test}/<scene>/images/*.png   (image-goal navigation)

The ``mini_test`` download split contains three test scenes and extracts into the original test paths.
"""
import hashlib
import json
import os
import tarfile

DATASET_REPO = os.environ.get("UNINAVBENCH_REPO", "yhncao/UniNavBench")
MANIFEST = "manifest.json"
STATE_FILE = ".uninavbench_shards.json"
SPLITS = ("train", "val", "test", "mini_test")


def default_data_dir():
    """$UNINAVBENCH_DATA, else ~/.cache/uninavbench/data (where ``download`` puts the dataset)."""
    if os.environ.get("UNINAVBENCH_DATA"):
        return os.environ["UNINAVBENCH_DATA"]
    return os.path.join(os.path.expanduser(os.environ.get("XDG_CACHE_HOME", "~/.cache")), "uninavbench", "data")


def _fetch(filename, source, repo_id, revision, from_dir):
    """Local path of ``filename`` from the dataset repository (downloaded to the hub cache if needed)."""
    if from_dir:
        path = os.path.join(from_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    if source == "hf":
        from huggingface_hub import hf_hub_download  # honours HF_ENDPOINT, e.g. a mirror
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
    if source == "modelscope":
        try:
            from modelscope.hub.file_download import dataset_file_download
        except ImportError as ex:
            raise ImportError("pip install 'uninavbench[modelscope]' to download from ModelScope") from ex
        return dataset_file_download(dataset_id=repo_id, file_path=filename, revision=revision or "master")
    raise ValueError(f"Unknown source: {source} (use 'hf' or 'modelscope')")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scene_of(member, split):
    parts = member.split("/")
    if parts[0] == split and len(parts) > 1:
        return parts[1]
    if parts[0] == "attributes" and len(parts) > 2 and parts[1] == split:
        return parts[2]
    return None


def download(splits=("test",), scenes=None, data_dir=None, source="hf", repo_id=DATASET_REPO, revision=None,
             from_dir=None, verify=True):
    """Downloads and extracts the shards of ``splits`` (optionally only ``scenes``). Returns the data dir."""
    data_dir = data_dir or default_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    with open(_fetch(MANIFEST, source, repo_id, revision, from_dir)) as f:
        manifest = json.load(f)
    wanted = None if scenes is None else {str(s) for s in scenes}
    for split in splits:
        available = {str(s) for shard in manifest["shards"] if shard["split"] == split
                     for s in shard["scenes"]}
        if not available:
            raise ValueError(f"Release does not contain split '{split}'")
        if wanted is not None and not wanted <= available:
            raise ValueError(f"Split '{split}' does not contain scenes {sorted(wanted - available)}")
    state_path = os.path.join(data_dir, STATE_FILE)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    for shard in manifest["shards"]:
        if shard["split"] not in splits:
            continue
        needed = set(shard["scenes"]) if wanted is None else set(shard["scenes"]) & wanted
        if not needed:
            continue
        previous = state.get(shard["file"], {})
        done = set(previous.get("scenes", [])) if previous.get("sha256") == shard["sha256"] else set()
        if needed <= done and state[shard["file"]].get("sha256") == shard["sha256"]:
            continue
        print(f"[uninavbench] {shard['file']} ({shard['bytes'] / 1e9:.2f} GB, scenes {sorted(needed, key=int)[:5]}...)", flush=True)
        path = _fetch(shard["file"], source, repo_id, revision, from_dir)
        if verify and _sha256(path) != shard["sha256"]:
            raise IOError(f"sha256 mismatch for {shard['file']}: remove it from the hub cache and retry")
        with tarfile.open(path) as tar:
            source_split = shard.get("source_split", shard["split"])
            members = [m for m in tar.getmembers() if _scene_of(m.name, source_split) in needed]
            if not members:
                raise IOError(f"No requested scenes found in {shard['file']}")
            for m in members:
                if m.name.startswith("/") or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
                    raise IOError(f"unsafe path in {shard['file']}: {m.name}")
            tar.extractall(data_dir, members=members)
        state[shard["file"]] = {"sha256": shard["sha256"], "scenes": sorted(done | needed, key=int)}
        with open(state_path + ".tmp", "w") as f:
            json.dump(state, f)
        os.replace(state_path + ".tmp", state_path)
    os.makedirs(os.path.join(data_dir, "attributes"), exist_ok=True)
    return data_dir

