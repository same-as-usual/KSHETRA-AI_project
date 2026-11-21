# India district boundaries (optional)

The Atlas accepts a GeoJSON FeatureCollection with `state` and `district` name properties. Upload it in the Atlas tab or place a file as `data/geo/india_districts.geojson`.

A public source is the [DataMeet India district boundaries project](https://projects.datameet.org/maps/districts/). It is released under CC BY 2.5 India and requires attribution. Its own documentation warns that several states have pre-delimitation boundaries, that there is some spatial shift, and that Telangana names may still appear under Andhra Pradesh. See the source page before use; these boundaries should not be used for legal or official decisions.

For matching, the app uses normalized state and district names. Unmatched features are left out rather than guessed. Map vintage is not the same as NFHS-5 (2019-21) geography for all districts.
