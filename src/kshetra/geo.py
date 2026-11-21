"""GeoJSON helpers for the district Atlas; no remote map tiles are embedded in project code."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from .text import norm_name, norm_state


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
