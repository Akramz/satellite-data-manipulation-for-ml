"""Acquires the tile image to be used as input to the ML pipeline."""

import requests
from pystac.extensions.eo import EOExtension as eo
import pystac_client
import planetary_computer
from shapely.geometry import mapping
import geopandas as gpd


def main(roi_path, output_path, start, end):

    # Search against the Planetary Computer STAC API
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Load the region of interest into a dictionary
    aoi = mapping(gpd.read_file(roi_path).geometry[0])

    # Define your temporal range
    daterange = {"interval": [start, end]}

    # Define your search with CQL2 syntax
    search = catalog.search(filter_lang="cql2-json", filter={
      "op": "and",
      "args": [
        {"op": "s_contains", "args": [{"property": "geometry"}, aoi]},
        {"op": "anyinteracts", "args": [{"property": "datetime"}, daterange]},
        {"op": "=", "args": [{"property": "collection"}, "sentinel-2-l2a"]}
      ]
    })

    # Get the search results
    items = search.item_collection()

    # Filter for the least cloudy result
    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)

    # Download the item
    response = requests.get(least_cloudy_item.assets["visual"].href, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to the output file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"TIFF file downloaded successfully as {output_path}")
    else:
        print(f"Error {response.status_code}: Unable to download the TIFF file")


if __name__ == "__main__":

    import argparse

    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_path", type=str, default="./data/roi.geojson")
    parser.add_argument("--output_path", type=str, default="./data/tile.tif")
    parser.add_argument("--start", type=str, default="2021-01-01T10:25:31Z")
    parser.add_argument("--end", type=str, default="2022-01-01T23:59:59Z")
    args = parser.parse_args()
    main(args.roi_path, args.output_path, args.start, args.end)