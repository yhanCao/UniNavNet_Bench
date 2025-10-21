from procthor_challenge.challenge import ProcthorChallenge
import argparse
import importlib
import gzip
import json
import logging
logging.getLogger().setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Inference script for RoboThor ObjectNav challenge.")

    parser.add_argument(
        "--agent", "-a",
        default="agents.random_agent",
        help="Relative module for agent definition.",
    )

    parser.add_argument(
        "--cfg", "-c",
        default="./configs/procthor.yaml",
        help="Filepath to challenge config.",
    )
    parser.add_argument(
        "--data_pth", "-d",
        default="dataset/Dynamic_Strategy",
        help="Filepath to challenge dataset.",
    )
    parser.add_argument(
        "--data_split", "-p",
        default="train",
        help="Filepath to challenge dataset.",
    )
    parser.add_argument(
        "--dynamic_level", "-l",
        default="small",
        help="Filepath to challenge dataset.",
    )
    parser.add_argument(
        "--scene_id", "-i",
        default="1",
        help="Filepath to challenge dataset.",
    )
    parser.add_argument(
        "--task_type", "-t",
        default="nav",
        help="Filepath to challenge results.",
    )
    parser.add_argument(
        "--start_episode_id", "-es",
        default=0,
        help="Filepath to challenge dataset.",
    )
    parser.add_argument(
        "--end_episode_id", "-ee",
        default=1,
        help="Filepath to challenge dataset.",
    )
    parser.add_argument(
        "--submission",
        action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true")
    parser.add_argument(
        "--train",
        action="store_true")
    parser.add_argument(
        "--val",
        action="store_true")
    parser.add_argument(
        "--test",
        action="store_true")

    parser.add_argument(
        "--nprocesses", "-n",
        default=1,
        type=int,
        help="Number of parallel processes used to compute inference.",
    )
    parser.add_argument(
        "--output", "-o",
        default="random_metrics.json.gz",
        help="Filepath to output results to.",
    )

    args = parser.parse_args()
    if args.submission:
        args.debug = False
        args.train = False
        args.val = True
        args.test = True

    agent = importlib.import_module(args.agent)
    agent_class, agent_kwargs, render_depth = agent.build()

    r = ProcthorChallenge(args.cfg, agent_class, agent_kwargs, args.data_pth, render_depth=render_depth)

    challenge_metrics = {}

    val_episodes = r.load_split(args.data_pth, args.data_split, args.dynamic_level, args.scene_id,args.start_episode_id,args.end_episode_id,args.task_type)
    challenge_metrics["val"] = r.inference(
        val_episodes,
        nprocesses=args.nprocesses,
        test=False,
        episode_id = args.start_episode_id
        )
    with gzip.open(args.output, "wt", encoding="utf-8") as zipfile:
        json.dump(challenge_metrics, zipfile)


if __name__ == "__main__":
    main()
