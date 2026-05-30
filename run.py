#!/usr/bin/env python3
import argparse
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

from benchmark import main

main(argparse.Namespace(
    model            = "poolside/Laguna-XS.2",
    budget           = 1024,
    n                = 100,
    humaneval        = True,
    livecodebench    = True,
    longbench        = True,
    longbench_tasks  = ["lcc", "repobench-p", "2wikimqa", "hotpotqa"],
    out              = "results/benchmark.json",
    baseline_only    = False,
))
