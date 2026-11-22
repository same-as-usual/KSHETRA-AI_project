# India district boundaries (optional)

The Atlas accepts a GeoJSON FeatureCollection with `state` and `district` name properties. Upload it in the Atlas tab or place a file as `data/geo/india_districts.geojson`.

A public source is the [DataMeet India district boundaries project](https://projects.datameet.org/maps/districts/). It is released under CC BY 2.5 India and requires attribution. Its own documentation warns that several states have pre-delimitation boundaries, that there is some spatial shift, and that Telangana names may still appear under Andhra Pradesh. See the source page before use; these boundaries should not be used for legal or official decisions.

For matching, the app uses normalized state and district names. Unmatched features are left out rather than guessed. Map vintage is not the same as NFHS-5 (2019-21) geography for all districts.


## Checking a boundary file before bundling

```bash
python scripts/check_district_geojson.py boundaries.geojson --district-prop DISTRICT --state-prop ST_NM --max-mb 3
```

The report gives file size, feature count, exact normalized name matches with the 707 profiles, and unmatched features. No polygon file is included in the repository. Unmatched or missing districts stay out of the choropleth and are not guessed.
