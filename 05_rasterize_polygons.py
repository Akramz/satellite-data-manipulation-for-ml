"""Produces the output masks used as targets for the model."""
import argparse
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import mapping
import geopandas as gpd


def rasterize_label_geoms(ref_file, gdf, output_file):
    """Rasterizes the geometries of a GeoDataFrame to create a label file.

    Args:
        ref_file (str): Path to the reference TIFF file to get the extents from.
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the geometries to be rasterized.
        output_file (str): Path to save the output TIFF file.
    """

    # Load the reference TIFF to get its extents
    with rio.open(ref_file) as src:
        out_shape = src.shape
        out_transform = src.transform
        out_crs = src.crs

    # Convert the polygon geometries to a list of GeoJSON-like dicts
    shapes = [(mapping(geom), value) for geom, value in gdf[["geometry", "y"]].values]

    # Rasterize the polygon geometries into a multi-band TIFF
    mask = rasterize(shapes,
                     out_shape=out_shape,
                     transform=out_transform,
                     dtype=rio.uint8,
                     fill=0,
                     default_value=0,
                     all_touched=True)

    # Save the output TIFF with the same extent, transform, and CRS as the reference TIFF
    with rio.open(output_file,
                  "w",
                  driver='GTiff',
                  height=out_shape[0],
                  width=out_shape[1],
                  dtype=rio.uint8,
                  crs=out_crs,
                  count=1,
                  transform=out_transform) as dst:
        dst.write(mask, indexes=1)


def main():

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--buildings_path", help="Path to the data file", default="./data/nairobi_buildings.geojson")
    parser.add_argument("--x_train_path", help="Path to the X_train.tif file", default="./data/X_train.tif")
    parser.add_argument("--x_val_path", help="Path to the X_val.tif file", default="./data/X_val.tif")
    parser.add_argument("--y_train_path", help="Path to the y_train.tif file", default="./data/y_train.tif")
    parser.add_argument("--y_val_path", help="Path to the y_val.tif file", default="./data/y_val.tif")

    args = parser.parse_args()

    # Load the buildings
    buildings = gpd.read_file(args.buildings_path).to_crs(rio.open(args.x_train_path).crs)

    # Rasterize the buildings to create the y_train.tif file
    rasterize_label_geoms(args.x_train_path, buildings, args.y_train_path)

    # Rasterize the buildings to create the y_val.tif file
    rasterize_label_geoms(args.x_val_path, buildings, args.y_val_path)
