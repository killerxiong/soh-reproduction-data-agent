import argparse
import json
from pathlib import Path

from config import Config
from utils import set_seed, ensure_dirs
from dataset_generator import generate_dataset
from trainer import train_model
from evaluator import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Executable reproduction pipeline for Wang 2024 PINN battery SOH paper")
    parser.add_argument("--plan", type=str, required=False, default=None, help="Path to normalized plan JSON")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    plan = None
    if args.plan:
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = json.load(f)

    cfg = Config(out_dir=args.out_dir, plan=plan)
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    generate_dataset(cfg)
    train_model(cfg)
    evaluate_model(cfg)

    print(f"Pipeline completed. Outputs written to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
