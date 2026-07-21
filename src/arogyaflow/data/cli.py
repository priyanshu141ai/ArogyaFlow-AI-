import argparse
from datetime import datetime
from pathlib import Path

from arogyaflow.data.generation import ScenarioConfig, ScenarioName
from arogyaflow.data.pipeline import run_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(ScenarioName))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--start", required=True, type=datetime.fromisoformat)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--anomaly-rate", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = ScenarioConfig(
        scenario=args.scenario,
        seed=args.seed,
        start=args.start,
        days=args.days,
        anomaly_rate=args.anomaly_rate,
    )
    run_generation(config, args.output)


if __name__ == "__main__":
    main()
