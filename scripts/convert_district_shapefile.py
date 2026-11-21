"""Convert a sourced district shapefile to the GeoJSON accepted by the Atlas.

Usage:
  python scripts/convert_district_shapefile.py path/to/districts.shp data/geo/india_districts.geojson

GeoPandas is an optional dependency listed in requirements-geo.txt.
Before using a boundary file, read data/geo/README.md and preserve its attribution/vintage.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shapefile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state-column", default="ST_NM")
    parser.add_argument("--district-column", default="DISTRICT")
    args = parser.parse_args()

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit("GeoPandas is optional; install dependencies listed in requirements-geo.txt") from exc

    frame = gpd.read_file(args.shapefile)
    for col in (args.state_column, args.district_column):
        if col not in frame.columns:
            raise SystemExit(f"Column {col!r} not found. Available columns: {', '.join(map(str, frame.columns))}")
    if frame.crs is None:
        raise SystemExit("The shapefile has no declared coordinate reference system; set it correctly before conversion.")
    if frame.crs.to_epsg() != 4326:
        frame = frame.to_crs(epsg=4326)
    frame = frame[[args.state_column, args.district_column, frame.geometry.name]].rename(
        columns={args.state_column: "state", args.district_column: "district"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(args.output, driver="GeoJSON")
    print(f"Wrote {len(frame)} features to {args.output}")
    print("Verify output state/district fields and attribution before loading it in the Atlas.")


if __name__ == "__main__":
    main()
