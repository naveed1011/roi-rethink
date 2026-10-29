#!/usr/bin/env python3
"""Run the two-turn agent over a dataset and write eval metrics + records.

Usage (offline demo with MockVLM):
    python scripts/run_eval.py --n 10 --mock --out results/demo.json

Usage with a real OpenAI-compatible VLM:
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=...   # optional, for local vLLM / Ollama / etc.
    python scripts/run_eval.py --n 20 --model gpt-4o-mini --out results/eval.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roi_rethink.agent import RethinkAgent  # noqa: E402
from roi_rethink.eval import evaluate, save_eval  # noqa: E402
from roi_rethink.synthetic import SyntheticAnomalyDataset  # noqa: E402
from roi_rethink.vlm import MockVLM, OpenAICompatibleVLM  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--mock", action="store_true", help="use offline MockVLM (no API key)")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--out", type=str, default="results/eval.json")
    p.add_argument("--quiet", action="store_true", help="disable the tqdm progress bar")
    args = p.parse_args()

    if args.mock:
        vlm = MockVLM(hint_bbox=(110, 110, 170, 170))
    else:
        vlm = OpenAICompatibleVLM(model=args.model)

    agent = RethinkAgent(vlm)
    samples = SyntheticAnomalyDataset(seed=args.seed).generate(args.n)
    result = evaluate(agent, samples, iou_threshold=args.iou_threshold, verbose=not args.quiet)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_eval(result, args.out)
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
