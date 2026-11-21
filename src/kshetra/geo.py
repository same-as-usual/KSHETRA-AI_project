"""GeoJSON and state-centroid helpers for the district Atlas; no remote map tiles are embedded in project code."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .text import norm_name, norm_state

# Reference geographic centers (latitude, longitude) for India's states and union territories.
# Used for contextual spatial plots when a full district polygon GeoJSON is not loaded.
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "ANDAMAN AND NICOBAR ISLANDS": (11.7401, 92.6586),
    "ANDHRA PRADESH": (15.9129, 79.7400),
    "ARUNACHAL PRADESH": (28.2180, 94.7278),
    "ASSAM": (26.2006, 92.9376),
    "BIHAR": (25.0961, 85.3131),
    "CHANDIGARH": (30.7333, 76.7794),
    "CHHATTISGARH": (21.2787, 81.8661),
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": (20.3974, 72.8328),
    "GOA": (15.2993, 74.1240),
    "GUJARAT": (22.2587, 71.1924),
    "HARYANA": (29.0588, 76.0856),
    "HIMACHAL PRADESH": (31.1048, 77.1734),
    "JAMMU AND KASHMIR": (33.7782, 76.5762),
    "JHARKHAND": (23.6102, 85.2799),
    "KARNATAKA": (15.3173, 75.7139),
    "KERALA": (10.8505, 76.2711),
    "LADAKH": (34.1526, 77.5771),
    "LAKSHADWEEP": (10.5667, 72.6417),
    "MADHYA PRADESH": (22.9734, 78.6569),
    "MAHARASHTRA": (19.7515, 75.7139),
    "MANIPUR": (24.6637, 93.9063),
    "MEGHALAYA": (25.4670, 91.3662),
    "MIZORAM": (23.1645, 92.9376),
    "NAGALAND": (26.1584, 94.5624),
    "NCT OF DELHI": (28.7041, 77.1025),
    "DELHI": (28.7041, 77.1025),
    "ODISHA": (20.9517, 85.0985),
    "PUDUCHERRY": (11.9416, 79.8083),
    "PUNJAB": (31.1471, 75.3412),
    "RAJASTHAN": (27.0238, 74.2179),
    "SIKKIM": (27.5330, 88.5122),
    "TAMIL NADU": (11.1271, 78.6569),
    "TELANGANA": (18.1124, 79.0193),
    "TRIPURA": (23.9408, 91.9882),
    "UTTAR PRADESH": (26.8467, 80.9462),
    "UTTARAKHAND": (30.0668, 79.0193),
    "WEST BENGAL": (22.9868, 87.8550),
}


def attach_approximate_coordinates(profiles: pd.DataFrame) -> pd.DataFrame:
    """Attach approximate display coordinates anchored to state centroids with deterministic spiral offsets.

    Important: These coordinates are for schematic state-grouped spatial visualization when
    official district boundary polygons are not uploaded; they are NOT survey GPS points or
    surveyed administrative centroids.
    """
    out = profiles.copy()
    if "state_key" not in out:
        out["state_key"] = out["state_raw"].map(norm_state)
    if "district_key" not in out:
        out["district_key"] = out["district_raw"].map(norm_name)

    lats = np.full(len(out), np.nan, dtype=float)
    lons = np.full(len(out), np.nan, dtype=float)

    for state_key, group in out.groupby("state_key", sort=False):
        center = STATE_CENTROIDS.get(str(state_key))
        if center is None:
            continue
        base_lat, base_lon = center
        idx = group.sort_values("district_key").index.to_list()
        n = len(idx)
        for rank, row_idx in enumerate(idx):
            if n == 1:
                lats[out.index.get_loc(row_idx)] = base_lat
                lons[out.index.get_loc(row_idx)] = base_lon
            else:
                # Golden-angle spiral within state radius (~0.18 to 1.15 deg) for non-overlapping display
                angle = rank * 2.399963229728653
                radius = 0.16 * np.sqrt(rank + 1)
                pos = out.index.get_loc(row_idx)
                lats[pos] = base_lat + radius * np.sin(angle) * 0.85
                lons[pos] = base_lon + radius * np.cos(angle)

    out["approx_lat"] = np.round(lats, 4)
    out["approx_lon"] = np.round(lons, 4)
    return out


def match_geojson_to_profiles(geojson: dict[str, Any], profiles: pd.DataFrame,
                              district_property: str, state_property: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Add a stable feature key to GeoJSON and return a matched profile subset.

    This is exact name matching after normalization only. Boundary vintages and district
    splits can differ from NFHS-5; unmatched polygons are excluded and reported by the caller.
    """
    if geojson.get("type") != "FeatureCollection" or not isinstance(geojson.get("features"), list):
        raise ValueError("GeoJSON must be a FeatureCollection with a features array")
    out_geo = deepcopy(geojson)
    keys: set[str] = set()
    valid = []
    for i, feature in enumerate(out_geo["features"]):
        props = feature.get("properties")
        if not isinstance(props, dict):
            props = {}
            feature["properties"] = props
        district = props.get(district_property)
        state = props.get(state_property)
        if district is None or state is None:
            props["_kshetra_key"] = None
            continue
        key = f"{norm_state(state)}::{norm_name(district)}"
        props["_kshetra_key"] = key
        keys.add(key)
        valid.append((key, state, district))

    profile = profiles.copy()
    if "state_key" not in profile:
        profile["state_key"] = profile["state_raw"].map(norm_state)
    if "district_key" not in profile:
        profile["district_key"] = profile["district_raw"].map(norm_name)
    profile["_kshetra_key"] = profile["state_key"].map(norm_state) + "::" + profile["district_key"].map(norm_name)
    matched = profile[profile["_kshetra_key"].isin(keys)].copy()
    out_geo["features"] = [f for f in out_geo["features"] if f.get("properties", {}).get("_kshetra_key") in set(matched["_kshetra_key"])]
    return out_geo, matched
