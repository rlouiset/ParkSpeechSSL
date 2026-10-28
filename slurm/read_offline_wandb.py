"""
Print the metrics of offline wandb runs straight from their run-*.wandb files, without
syncing (compute nodes on Jean Zay log with WANDB_MODE=offline). Works on runs that are
still being written.

    python3 slurm/read_offline_wandb.py wandb/offline-run-20261028_*            # last HC/PD + clinical values
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
]
# clinical_probe.py probes, each logged as Clinical/{name} (pooled out-of-fold score) plus
# _fold_mean / _fold_std / _n. Spearman rho for the regressions, AUC for sex.
CLINICAL_PROBES = [
    "updrs_pd", "hy_pd", "age_pd", "age_hc", "sex_pd", "sex_hc", "moca_pd", "moca_hc", "moca",
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


def print_clinical(rows: list[dict]) -> None:
    """Last clinical-probe epoch: pooled score, fold mean +/- fold std, n individuals."""
    clinical = [row for row in rows if any(k.startswith("Clinical/") for k in row)]
    if not clinical:
        return
    last = clinical[-1]
    epoch = last.get("epoch", "?")
    print(f"  clinical probes (epoch {epoch}): pooled  |  fold mean +/- std  |  n")
    for name in CLINICAL_PROBES:
        key = f"Clinical/{name}"
        if key not in last:
            continue
        mean, std, n = (last.get(f"{key}_{s}", math.nan) for s in ("fold_mean", "fold_std", "n"))
        extra = f"  MAE {last[key + '_mae']:.2f}" if f"{key}_mae" in last else ""
        print(f"    {name:10s} {fmt(last[key]):>8s}  |  {fmt(mean)} +/- {fmt(std)}  |  {int(n) if n == n else '?'}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path, help="offline-run-* directories")
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="metrics to show (default: HC/PD, then every clinical probe)")
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
            if args.keys == DEFAULT_KEYS:
                print_clinical(rows)


if __name__ == "__main__":
    main()
