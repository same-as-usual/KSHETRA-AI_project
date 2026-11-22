"""Check a district boundary GeoJSON before placing it at data/geo/india_districts.geojson.

Example:
    python scripts/check_district_geojson.py boundaries.geojson --district-prop DISTRICT --state-prop ST_NM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kshetra.geo import audit_district_geojson  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("geojson", type=Path)
    parser.add_argument("--profiles", type=Path, default=ROOT / "outputs" / "district_profiles.csv")
    parser.add_argument("--district-prop", required=True)
    parser.add_argument("--state-prop", required=True)
    parser.add_argument("--max-mb", type=float, default=3.0)
    args = parser.parse_args()

    raw = args.geojson.read_bytes()
    try:
        geojson = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"{args.geojson} is not valid JSON: {exc}")
    report = audit_district_geojson(geojson, pd.read_csv(args.profiles), args.district_prop,
                                    args.state_prop, file_bytes=len(raw), max_mb=args.max_mb)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["within_size_limit"] is False:
        sys.exit("File exceeds the size limit; simplify geometry before bundling.")


if __name__ == "__main__":
    main()
