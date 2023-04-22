"""Crops the original tile to the `train` & `validation` regions of interest and create the `X_train.tif` & `X_val.tif` files."""
import argparse
import pyproj
import rasterio as rio
from rasterio.mask import mask
from shapely.ops import transform
import geopandas as gpd


def crop_tile_save(tile_path, roi, output_path):
    """Crops the `tile_path` image to the `roi` and saves it to `output_path`.

    Args:
        tile_path (str): Path to the tile image to be cropped.
        roi (shapely.geometry.Polygon): The region of interest we want to use to crop.
        output_path (str): Path to save the cropped image.
    """

    # Get the CRS of the tile
    crs = rio.open(tile_path).crs

    # Create a Transformer object to project the geometry into the CRS of the tile
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Define a function to apply the transformer to the geometry
    def transform_coords(transformer, geom):
        """Transforms the coordinates of a geometry using the `transformer` object.

        Args:
            transformer (pyproj.Transformer): The transformer object.
            geom (shapely.geometry.Polygon): The geometry to be transformed.

        Returns:
            shapely.geometry.Polygon: The transformed geometry.
        """
        return transform(lambda x, y: transformer.transform(x, y), geom)

    # Project the geometry into the new CRS
    roi = transform_coords(transformer, roi)

    # Crop and save
    with rio.open(tile_path) as src:
        # Mask the TIFF using the Polygon
        out_image, out_transform = mask(src, [roi], crop=True)

        # Save the cropped TIFF
        with rio.open(output_path,
                      'w',
                      driver='GTiff',
                      height=out_image.shape[1],
                      width=out_image.shape[2],
                      count=src.count,
                      dtype=out_image.dtype,
                      crs=src.crs,
                      transform=out_transform) as dst:
            dst.write(out_image)


def main():

    # Get the below paths from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the input tile", default="./data/tile.tif")
    parser.add_argument("--train_roi_path", help="Path to the train region of interest file",
                        default="./data/train_roi.gpkg")
    parser.add_argument("--val_roi_path", help="Path to the validation region of interest file",
                        default="./data/val_roi.gpkg")
    parser.add_argument("--xtrain_path", help="Path to the output cropped tile for the train region of interest",
                        default="./data/X_train.tif")
    parser.add_argument("--xval_path", help="Path to the output cropped tile for the validation region of interest",
                        default="./data/X_val.tif")

    # Set the paths
    args = parser.parse_args()
    intput_path = args.input_path
    train_roi_path = args.train_roi_path
    val_roi_path = args.val_roi_path
    xtrain_path = args.xtrain_path
    xval_path = args.xval_path

    # Load the train and validation regions of interest
    train_roi = gpd.read_file(train_roi_path).geometry[0]
    val_roi = gpd.read_file(val_roi_path).geometry[0]

    # Crop the tile to the train region of interest
    crop_tile_save(intput_path, train_roi, xtrain_path)

    # Crop the tile to the validation region of interest
    crop_tile_save(intput_path, val_roi, xval_path)