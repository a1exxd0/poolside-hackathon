#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("HF_TOKEN"):
    sys.exit("ERROR: HF_TOKEN not found in .env")

os.environ.setdefault(
    "HF_HOME",
    str(Path(os.environ.get("EPHEMERAL", Path.home())) / ".cache" / "huggingface"),
)

Path("results").mkdir(exist_ok=True)

sys.argv = [
    "benchmark.py",
    "--model",            "poolside/Laguna-XS.2",
    "--budget",           "1024",
    "--humaneval",
    "--livecodebench",
    "--longbench",
    "--longbench-tasks",  "lcc", "repobench-p", "2wikimqa", "hotpotqa",
    "--n",                "100",
    "--out",              "results/benchmark.json",
]

from benchmark import main
main()
