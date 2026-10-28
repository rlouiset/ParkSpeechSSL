"""
Print the metrics of offline wandb runs straight from their run-*.wandb files, without
syncing (compute nodes on Jean Zay log with WANDB_MODE=offline). Works on runs that are
still being written.

    python3 slurm/read_offline_wandb.py wandb/offline-run-20261028_*            # last value of each metric
    python3 slurm/read_offline_wandb.py wandb/offline-run-20261028_* --history  # one row per logged step
    python3 slurm/read_offline_wandb.py wandb/offline-run-* --keys Val/hc_pd_auc Clinical/moca
"""

import argparse
import json
import math
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

DEFAULT_KEYS = [
    "epoch",
    "Val/hc_pd_balanced_accuracy",
    "Val/hc_pd_auc",
    "Val/hc_pd_balanced_accuracy_individual",
    "Val/hc_pd_auc_individual",
    "Clinical/moca",
    "Clinical/moca_pd",
    "Clinical/moca_hc",
]


def read_run(run_dir: Path) -> tuple[str, list[dict]]:
    """(wandb run name, history rows) of one offline-run-* directory."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(next(run_dir.glob("run-*.wandb"))))
    name, rows = run_dir.name, []
    while True:
        try:
            data = ds.scan_data()
        except AssertionError:  # last record still being written by a running job
            break
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        kind = record.WhichOneof("record_type")
        if kind == "run" and record.run.display_name:
            name = record.run.display_name
        elif kind == "history":
            row = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                row[key] = json.loads(item.value_json)
            rows.append(row)
    return name, rows


def fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path, help="offline-run-* directories")
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="metrics to show (default: HC/PD + MoCA)")
    parser.add_argument("--history", action="store_true", help="every logged row, not just the last value")
    parser.add_argument("--list-keys", action="store_true", help="print every metric name logged in the run")
    args = parser.parse_args()

    for run_dir in args.runs:
        name, rows = read_run(run_dir)
        print(f"\n=== {name} ({run_dir.name}, {len(rows)} rows)")
        if args.list_keys:
            print("\n".join(sorted({k for row in rows for k in row if not k.startswith("_")})))
            continue
        if args.history:
            for row in rows:
                shown = {k: row[k] for k in args.keys if k in row}
                if any(k != "epoch" for k in shown):
                    print("  " + "  ".join(f"{k}={fmt(v)}" for k, v in shown.items()))
        else:
            last = {}
            for row in rows:
                last.update({k: v for k, v in row.items() if k in args.keys})
            for key in args.keys:
                if key in last:
                    print(f"  {key:45s} {fmt(last[key])}")
            best = [row["Val/hc_pd_balanced_accuracy"] for row in rows if row.get("Val/hc_pd_balanced_accuracy", -1) >= 0]
            if best:
                print(f"  {'best Val/hc_pd_balanced_accuracy':45s} {max(best):.4f}")


if __name__ == "__main__":
    main()
