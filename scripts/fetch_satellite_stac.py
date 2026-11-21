"""Optional STAC / satellite metadata & aggregation helper for KSHETRA-AI Module 2.

Queries Microsoft Planetary Computer's public STAC API (no API key required for catalog search)
or transforms local raster/zonal statistics into the verified CSV format expected by
``src/kshetra/environment.py`` (`state,district,observed_at,source,rainfall_deficit_pct,ndvi_change_pct,viirs_change_pct,fire_count`).

Usage:
    python scripts/fetch_satellite_stac.py --state "Bihar" --district "Araria" --bbox 87.1 25.9 87.7 26.5 --month 2025-07
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def query_stac_scenes(bbox: list[float], date_range: str, collection: str = "sentinel-2-l2a", limit: int = 5) -> list[dict[str, Any]]:
    """Query the Planetary Computer STAC catalog for available scenes over a district bounding box."""
    payload = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": date_range,
        "limit": limit,
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(STAC_SEARCH_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    features = data.get("features", [])
    return [
        {
            "id": f.get("id"),
            "datetime": (f.get("properties") or {}).get("datetime"),
            "cloud_cover": (f.get("properties") or {}).get("eo:cloud_cover"),
            "platform": (f.get("properties") or {}).get("platform"),
        }
        for f in features
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query STAC satellite scene provenance for KSHETRA-AI Module 2.")
    parser.add_argument("--state", required=True, help="State name (e.g. Bihar)")
    parser.add_argument("--district", required=True, help="District name (e.g. Araria)")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--month", required=True, help="Target YYYY-MM (e.g. 2025-07)")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save scene metadata JSON")
    args = parser.parse_args()

    date_range = f"{args.month}-01T00:00:00Z/{args.month}-28T23:59:59Z"
    scenes = query_stac_scenes(args.bbox, date_range)
    report = {
        "state": args.state,
        "district": args.district,
        "bbox": args.bbox,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "collection": "sentinel-2-l2a (Microsoft Planetary Computer STAC)",
        "scenes_found": len(scenes),
        "scenes": scenes,
        "note": "Compute zonal NDVI/CHIRPS/VIIRS anomalies from the matched rasters before exporting to environment_signals.csv.",
    }
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
