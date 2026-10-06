# -*- coding: utf-8 -*-
"""Kalman sweep sonucunu ozetle ve en iyi parametre adayini sec."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


BASELINE = "off_visual_no_revert"


def f(value, default=math.nan):
    try:
        return float(value)
    except Exception:
        return default


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(values):
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def score_row(row: dict) -> tuple[float, float, float]:
    # Ana hedef RMSE dusurmek; esitlige yakin durumda dogruluk ve F skoru belirleyici.
    return (
        f(row.get("rmse_km"), math.inf),
        -f(row.get("accuracy"), 0.0),
        -f(row.get("f_skoru"), 0.0),
    )


def analyze(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if int(f(r.get("returncode"), 1)) == 0 and f(r.get("total"), 0) > 0]
    by_route = defaultdict(list)
    by_config = defaultdict(list)
    for row in ok_rows:
        by_route[row["route"]].append(row)
        by_config[row["config"]].append(row)

    route_summaries = []
    for route, items in sorted(by_route.items()):
        baseline = next((r for r in items if r["config"] == BASELINE), None)
        kalman_items = [r for r in items if r["config"] != BASELINE]
        best_kalman = min(kalman_items, key=score_row) if kalman_items else None
        best_any = min(items, key=score_row)
        summary = {
            "route": route,
            "baseline_rmse_m": 1000.0 * f(baseline.get("rmse_km")) if baseline else None,
            "baseline_accuracy": f(baseline.get("accuracy")) if baseline else None,
            "best_any_config": best_any["config"],
            "best_any_rmse_m": 1000.0 * f(best_any.get("rmse_km")),
        }
        if best_kalman:
            base_rmse = f(baseline.get("rmse_km")) if baseline else math.nan
            kalman_rmse = f(best_kalman.get("rmse_km"))
            summary.update({
                "best_kalman_config": best_kalman["config"],
                "best_kalman_rmse_m": 1000.0 * kalman_rmse,
                "best_kalman_accuracy": f(best_kalman.get("accuracy")),
                "kalman_delta_rmse_m": 1000.0 * (kalman_rmse - base_rmse) if math.isfinite(base_rmse) else None,
                "kalman_better_than_baseline": bool(math.isfinite(base_rmse) and kalman_rmse < base_rmse),
            })
        route_summaries.append(summary)

    config_summaries = []
    route_count = len(by_route)
    for config, items in sorted(by_config.items()):
        rmses = [1000.0 * f(r.get("rmse_km")) for r in items]
        accs = [f(r.get("accuracy"), 0.0) for r in items]
        fs = [f(r.get("f_skoru"), 0.0) for r in items]
        better = 0
        worse = 0
        for row in items:
            if config == BASELINE:
                continue
            baseline = next((r for r in by_route[row["route"]] if r["config"] == BASELINE), None)
            if not baseline:
                continue
            if f(row.get("rmse_km"), math.inf) < f(baseline.get("rmse_km"), math.inf):
                better += 1
            else:
                worse += 1
        config_summaries.append({
            "config": config,
            "routes": len(items),
            "complete": len(items) == route_count,
            "mean_rmse_m": mean(rmses),
            "median_rmse_m": sorted(rmses)[len(rmses) // 2] if rmses else math.nan,
            "mean_accuracy": mean(accs),
            "mean_f_skoru": mean(fs),
            "routes_better_than_baseline": better,
            "routes_not_better_than_baseline": worse,
        })
    config_summaries.sort(
        key=lambda r: (
            r["routes"] != route_count,
            r["mean_rmse_m"],
            -r["routes_better_than_baseline"],
            -r["mean_accuracy"],
        )
    )

    return {
        "route_count": route_count,
        "row_count": len(ok_rows),
        "best_global_config": config_summaries[0]["config"] if config_summaries else None,
        "configs": config_summaries,
        "routes": route_summaries,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    result = analyze(load_rows(args.csv_path))
    out_dir = args.csv_path.parent
    (out_dir / "kalman_sweep_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "kalman_sweep_config_summary.csv", result["configs"])
    write_csv(out_dir / "kalman_sweep_route_summary.csv", result["routes"])
    print(json.dumps({
        "route_count": result["route_count"],
        "row_count": result["row_count"],
        "best_global_config": result["best_global_config"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
