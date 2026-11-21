"""Optional STAC / satellite metadata & aggregation helper for KSHETRA-AI Module 2.

Queries Microsoft Planetary Computer's public STAC API (no API key required for catalog search)
and returns scene metadata only, not district measurements. The results are not an environmental
CSV and must not be used as alerts without a separate, validated raster/zonal-statistics pipeline.

Usage:
    python scripts/fetch_satellite_stac.py --state "Bihar" --district "Araria" --bbox 87.1 25.9 87.7 26.5 --month 2025-07
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def query_stac_scenes(bbox: list[float], date_range: str, collection: str = "sentinel-2-l2a", limit: int = 5, *, search_url: str = STAC_SEARCH_URL) -> list[dict[str, Any]]:
    """Query the Planetary Computer STAC catalog for available scenes over a district bounding box."""
    if len(bbox) != 4 or not all(map(math.isfinite, bbox)):
        raise ValueError("bbox must contain four finite coordinates")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("bbox min coordinates must be less than max coordinates")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    payload = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": date_range,
        "limit": limit,
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(search_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("STAC response must be a JSON object")
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError("STAC response has no features array")
    return [
        {
            "id": f.get("id"),
            "datetime": (f.get("properties") or {}).get("datetime"),
            "cloud_cover": (f.get("properties") or {}).get("eo:cloud_cover"),
            "platform": (f.get("properties") or {}).get("platform"),
        }
        for f in features if isinstance(f, dict)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query STAC satellite scene provenance for KSHETRA-AI Module 2.")
    parser.add_argument("--state", required=True, help="State name (e.g. Bihar)")
    parser.add_argument("--district", required=True, help="District name (e.g. Araria)")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--month", required=True, help="Target YYYY-MM (e.g. 2025-07)")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save scene metadata JSON")
    parser.add_argument("--search-url", default=STAC_SEARCH_URL, help="Full STAC item-search endpoint")
    args = parser.parse_args()

    try:
        month = datetime.strptime(args.month, "%Y-%m")
        last = calendar.monthrange(month.year, month.month)[1]
        date_range = f"{month:%Y-%m}-01T00:00:00Z/{month:%Y-%m}-{last:02d}T23:59:59Z"
        scenes = query_stac_scenes(args.bbox, date_range, search_url=args.search_url)
    except (httpx.HTTPError, ValueError) as exc:
        parser.exit(1, f"STAC search failed ({args.search_url}): {exc}\n")
    report = {
        "state": args.state,
        "district": args.district,
        "bbox": args.bbox,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "collection": "sentinel-2-l2a",
        "search_url": args.search_url,
        "datetime": date_range,
        "scenes_found": len(scenes),
        "scenes": scenes,
        "note": "Compute zonal NDVI/CHIRPS/VIIRS anomalies from the matched rasters before exporting to environment_signals.csv.",
    }
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
