"""Command line interface.

    uninavbench run   -a my_agent_module -d DATA -p test -l all -i 0 -t nav -es 0 -ee 2000
    uninavbench batch -a my_agent_module -d DATA -p test --levels static small all --tasks nav eqa -n 4
"""
import argparse
import gzip
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from .challenge import DEFAULT_CONFIG, ProcthorChallenge

DEFAULT_AGENT = "uninavbench.agents.random_agent"


def load_agent(module_name):
    """``module_name`` is an importable module (or a .py file) exposing build() -> (class, kwargs, render_depth)."""
    if module_name.endswith(".py") and os.path.isfile(module_name):
        spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(module_name))[0], module_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        sys.path.insert(0, os.getcwd())
        module = importlib.import_module(module_name)
    return module.build()


def add_common(parser):
    parser.add_argument("--agent", "-a", default=DEFAULT_AGENT, help="Agent module (or .py file) with a build() function.")
    parser.add_argument("--cfg", "-c", default=DEFAULT_CONFIG, help="Benchmark config (YAML).")
    parser.add_argument("--data_pth", "-d", default=os.environ.get("UNINAVBENCH_DATA", os.path.join("dataset", "Dynamic_Strategy")),
                        help="Dataset root containing train/ val/ test/ attributes/ (default: $UNINAVBENCH_DATA).")
    parser.add_argument("--data_split", "-p", default="test", choices=["train", "val", "test"])
    parser.add_argument("--start_episode_id", "-es", default=0, type=int)
    parser.add_argument("--end_episode_id", "-ee", default=2000, type=int)
    parser.add_argument("--result_path", "-r", default=None, help="Directory for per-episode results (config result_path).")


def run(args):
    agent_class, agent_kwargs, render_depth = load_agent(args.agent)
    bench = ProcthorChallenge(args.cfg, agent_class, agent_kwargs, args.data_pth, render_depth=render_depth)
    metrics = bench.evaluate(args.data_split, args.dynamic_level, args.scene_id, args.task_type,
                             args.start_episode_id, args.end_episode_id, output_dir=args.result_path)
    with gzip.open(args.output, "wt", encoding="utf-8") as f:
        json.dump({"val": metrics}, f)
    summary = {k: v for k, v in metrics.items() if k != "episodes"}
    print(json.dumps(summary, indent=2))


def batch(args):
    scenes = args.scenes or sorted(os.listdir(os.path.join(args.data_pth, args.data_split)), key=int)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    jobs = [(s, l, t) for s in scenes for l in args.levels for t in args.tasks]

    def one(job):
        scene, level, task = job
        name = f"{args.data_split}_{scene}_{level}_{task}"
        out = os.path.join(args.out, name + ".json.gz")
        if os.path.exists(out):
            return name, "skipped"
        cmd = [sys.executable, "-m", "uninavbench", "run", "-a", args.agent, "-c", args.cfg, "-d", args.data_pth,
               "-p", args.data_split, "-l", level, "-i", str(scene), "-t", task,
               "-es", str(args.start_episode_id), "-ee", str(args.end_episode_id), "-o", out + ".tmp"]
        if args.result_path:
            cmd += ["-r", args.result_path]
        with open(os.path.join(args.logs, name + ".log"), "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if code == 0:
            os.replace(out + ".tmp", out)
            return name, "done"
        return name, f"FAILED (exit {code}, see {args.logs}/{name}.log)"

    with ThreadPoolExecutor(args.nprocesses) as pool:
        for name, status in pool.map(one, jobs):
            print(name, status, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="uninavbench", description="UniNavBench: long-horizon dynamic navigation and Active-EQA.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Evaluate one scene, level and task family.")
    add_common(p_run)
    p_run.add_argument("--dynamic_level", "-l", default="all", choices=["static", "small", "all"])
    p_run.add_argument("--scene_id", "-i", default="0")
    p_run.add_argument("--task_type", "-t", default="nav", choices=["nav", "eqa"])
    p_run.add_argument("--output", "-o", default="metrics.json.gz", help="Metrics file (gzip JSON).")
    p_run.add_argument("--verbose", "-v", action="store_true")
    p_run.set_defaults(func=run)

    p_batch = sub.add_parser("batch", help="Evaluate every scene of a split (resumable).")
    add_common(p_batch)
    p_batch.add_argument("--levels", nargs="+", default=["static", "small", "all"])
    p_batch.add_argument("--tasks", nargs="+", default=["nav", "eqa"])
    p_batch.add_argument("--scenes", nargs="+", default=None, help="Scene ids (default: every scene of the split).")
    p_batch.add_argument("--nprocesses", "-n", default=1, type=int, help="Parallel simulator processes.")
    p_batch.add_argument("--out", default=os.path.join("Result", "metrics"))
    p_batch.add_argument("--logs", default=os.path.join("Result", "logs"))
    p_batch.set_defaults(func=batch, verbose=False)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
