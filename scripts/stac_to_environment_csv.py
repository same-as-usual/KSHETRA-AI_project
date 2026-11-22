"""Convert per-district zonal means plus STAC scene metadata into environment_signals CSV.

Inputs:
  --zonal-table   CSV with one row per district/date (see kshetra.zonal.ZONAL_REQUIRED).
                  Values come from clipping rasters to district polygons in an external GIS tool.
  --scenes-json   Output of scripts/fetch_satellite_stac.py (scene id and eo:cloud_cover).
Output:
  --out           environment_signals CSV accepted by the Module 2 uploader.

This script does not read rasters and does not download data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kshetra.zonal import zonal_table_to_signals  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zonal-table", type=Path, required=True)
    parser.add_argument("--scenes-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cloud", type=float, default=20.0, help="Maximum Sentinel-2 cloud cover (%%)")
    parser.add_argument("--rejected-out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.scenes_json.read_text(encoding="utf-8"))
    scenes = {s["id"]: s.get("cloud_cover") for s in report.get("scenes", []) if s.get("id")}
    table = pd.read_csv(args.zonal_table, dtype=str, keep_default_na=False)
    for col in ("ndvi_current", "ndvi_baseline", "viirs_current", "viirs_baseline", "rain_observed_mm", "rain_normal_mm", "fire_count"):
        if col in table:
            table[col] = pd.to_numeric(table[col].replace("", None), errors="coerce")
    signals, rejected = zonal_table_to_signals(table, scenes, max_cloud_cover=args.max_cloud)
    signals.to_csv(args.out, index=False)
    if args.rejected_out is not None:
        rejected.to_csv(args.rejected_out, index=False)
    print(f"wrote {len(signals)} rows to {args.out}; rejected {len(rejected)}")


if __name__ == "__main__":
    main()
