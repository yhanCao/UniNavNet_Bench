# Building and publishing the dataset (maintainers)

`scripts/build_release.py` packs the generated data into the tar shards and `manifest.json` that
`uninavbench download` fetches: per scene the `{nav,eqa}.json.gz` episodes of the three levels (compact
JSON), the `strategy_final_{small,all}.json` schedules and the goal images used by image-goal navigation,
25 scenes per shard, plus a `mini_test` shard with test scenes 0, 4 and 6.

```bash
python scripts/build_release.py /path/to/Dynamic_Strategy release/ --resume   # train/val/test + mini_test
huggingface-cli upload yhncao/UniNavBench release/ . --repo-type dataset
modelscope upload yhncao/UniNavBench release/ --repo-type dataset             # mirror
```

`--resume` reuses existing shards after checking their sha256. `UNINAVBENCH_REPO` (or `--repo` of
`uninavbench download`) selects another dataset repository.

Upload the changed shards, `manifest.json` and the dataset card in one commit, so that the manifest never
refers to shards that are not there yet, and compare the remote LFS sha256 of every shard with the
manifest afterwards. Fixes to released data are applied to the files themselves and summarized in the
dataset card ("Changes in this release"); see also [nav-220-229-repair.md](nav-220-229-repair.md).
