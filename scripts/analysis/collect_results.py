import re
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


RUN_RE = re.compile(
    r"(?P<protocol>.+)_c(?P<num_clients>\d+)_alpha(?P<alpha>[\d.]+)_seed(?P<seed>\d+)_r(?P<rounds>\d+)(?:_run(?P<run_id>.+))?$"
)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return data if data is not None else {}


def parse_run_name(run_dir: Path) -> dict:
    match = RUN_RE.match(run_dir.name)
    if not match:
        return {
            "protocol": run_dir.name,
            "num_clients": None,
            "alpha": None,
            "seed": None,
            "rounds": None,
        }

    out = match.groupdict()
    out["num_clients"] = int(out["num_clients"])
    out["alpha"] = float(out["alpha"])
    out["seed"] = int(out["seed"])
    out["rounds"] = int(out["rounds"])
    out["run_id"] = out.get("run_id")
    return out


def metric_dict_to_round_rows(metrics: dict, prefix: str) -> list[dict]:
    rows = []

    if not metrics:
        return rows

    max_len = max(len(v) for v in metrics.values() if isinstance(v, list))

    for round_idx in range(max_len):
        row = {"round": round_idx + 1}

        for key, values in metrics.items():
            if isinstance(values, list) and round_idx < len(values):
                row[f"{prefix}_{key}"] = values[round_idx]

        rows.append(row)

    return rows


def main():
    base = Path("outputs/creditcard")
    out_dir = Path("results/summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_round_rows = []
    final_rows = []

    for run_dir in sorted(base.glob("*")):
        if not run_dir.is_dir():
            continue

        meta = parse_run_name(run_dir)

        fit_metrics = load_yaml(run_dir / "fit_metrics.yaml")
        eval_metrics = load_yaml(run_dir / "eval_metrics.yaml")

        fit_rows = metric_dict_to_round_rows(fit_metrics, "fit")
        eval_rows = metric_dict_to_round_rows(eval_metrics, "eval")

        by_round = {}

        for row in fit_rows:
            by_round.setdefault(row["round"], {}).update(row)

        for row in eval_rows:
            by_round.setdefault(row["round"], {}).update(row)

        for round_num, row in sorted(by_round.items()):
            full_row = dict(meta)
            full_row["run_name"] = run_dir.name
            full_row.update(row)
            all_round_rows.append(full_row)

        if by_round:
            last_round = max(by_round)
            final_row = dict(meta)
            final_row["run_name"] = run_dir.name
            final_row.update(by_round[last_round])
            final_rows.append(final_row)

    round_df = pd.DataFrame(all_round_rows)
    final_df = pd.DataFrame(final_rows)

    round_path = out_dir / "round_metrics.csv"
    final_path = out_dir / "final_metrics.csv"

    round_df.to_csv(round_path, index=False)
    final_df.to_csv(final_path, index=False)

    print(f"Wrote {round_path} with {len(round_df)} rows")
    print(f"Wrote {final_path} with {len(final_df)} rows")

    if len(final_df) > 0:
        print(final_df.tail())


if __name__ == "__main__":
    main()