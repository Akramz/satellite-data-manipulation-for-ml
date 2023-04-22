"""Loads the polygons from the data file and filters the high-confidence ones that are inside the region of interest."""
import argparse
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads


def main(buildings_path, roi_path, output_path):

    # Load the buildings using pandas (after downloading & unzipping the file [check Step 3])
    df = pd.read_csv(buildings_path)

    # Filter for high-confidence buildings
    df = df[df["confidence"] > 0.83]

    # Get the geometries of the buildings and load them into objects (from strings)
    geometries = [loads(geom) for geom in df["geom"].tolist()]

    # Load the original ROI
    roi = gpd.read_file(roi_path).geometry[0]

    # Filter the geometries that are inside the region of interest
    nairobi_buildings = [geom for geom in geometries if roi.contains(geom)]

    # Mark the buildings as `1` and Save the file
    data = {"y": [1]*len(nairobi_buildings), "geometry": nairobi_buildings}
    gpd.GeoDataFrame(data=data, crs="EPSG:4326").to_file(output_path)


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--buildings_path", help="Path to the data file", default="./data/183_buildings.csv")
    parser.add_argument("--roi_path", help="Path to the region of interest file", default="./data/roi.geojson")
    parser.add_argument("--output_path", help="Path to the output file", default="./data/nairobi_buildings.geojson")

    args = parser.parse_args()

    # Run the main function
    main(args.buildings_path, args.roi_path, args.output_path)