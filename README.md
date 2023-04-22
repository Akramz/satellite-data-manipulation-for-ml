# Creating a Remote Sensing ML dataset

The goal of this guide is to outline the necessary steps to create a remote sensing dataset for ML applications (without going into ML). `TorchGeo` can be directly used on the datasets created by this guide, but it is not required.

Our approach is to demonstrate through an example. The problem we suggest has the following characteristics:

- We want to detect **buildings** in Africa using public satellite imagery.
- As input, we will use [Sentinel-2](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) imagery.
- As output, we will use Google's [Open Buildings](https://sites.research.google/open-buildings/) dataset.

The final output we want consist of the following files:

- `X_train.tif`: A 3-band (RGB) GeoTIFF file containing the training input Sentinel-2 imagery.
- `y_train.tif`: A 1-band GeoTIFF file containing the training output Open Buildings labels as a mask.
- `X_val.tif`: A 3-band (RGB) GeoTIFF file containing the validation input Sentinel-2 imagery.
- `y_val.tif`: A 1-band GeoTIFF file containing the validation output Open Buildings labels as a mask.

## Step 1: Define the Region of Interest (ROI)

We set the overall region of interest to be **Nairobi, Kenya**.

We create a Python function to generate the bounding box coordinates for recognizable city names:

```python
import geopandas as gpd
from shapely.geometry import Polygon
from geopy.geocoders import Nominatim

def city_bounding_box(city_name):
    """Get the bounding box coordinates for a city name.
    
    Args:
        city_name (str): The name of the city.
    
    Returns:
        shapely.geometry.Polygon: The bounding box coordinates.
    """
    
    geolocator = Nominatim(user_agent="city_bounding_box")
    location = geolocator.geocode(city_name)
    
    if location is None:
        raise ValueError(f"Unable to find the location for '{city_name}'")
    
    bbox = location.raw.get('boundingbox')
    
    if not bbox:
        raise ValueError(f"Unable to obtain the bounding box for '{city_name}'")
    
    min_lat, max_lat, min_lon, max_lon = map(float, bbox)
    
    # Create a shapely polygon using the bounding box coordinates
    polygon = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)])
    
    return polygon

# Get the train & validation ROIs
roi = city_bounding_box("Nairobi")

# Save it to a GeoJSON file
gpd.GeoDataFrame(geometry=[roi], crs="EPSG:4326").to_file("./data/roi.geojson")
```

## Step 2: Filter & Acquire the tile image for the ROI

We filter the Sentinel-2 imagery to get the least-cloudy tile for the ROI:

```python
import requests
from pystac.extensions.eo import EOExtension as eo
import pystac_client
import planetary_computer
from shapely.geometry import mapping
import geopandas as gpd

# Search against the Planetary Computer STAC API
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Load the region of interest into a dictionary
aoi = mapping(gpd.read_file("./data/roi.geojson").geometry[0]) 

# Define your temporal range
daterange = {"interval": ["2021-01-01T10:25:31Z", "2022-01-01T23:59:59Z"]}

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
output_file = "./data/tile.tif"
response = requests.get(least_cloudy_item.assets["visual"].href, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Save the content to the output file
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"TIFF file downloaded successfully as {output_file}")
else:
    print(f"Error {response.status_code}: Unable to download the TIFF file")
```

## Step 3: Acquire & filter the Open Buildings polygons for the ROI

Next, we need to download the Open Buildings dataset for a cell that contains Nairobi then filter the polygons further to only include those that intersect with the ROI:

1. Download the file from [here](https://storage.googleapis.com/open-buildings-data/v2/polygons_s2_level_4_gzip/183_buildings.csv.gz).
2. Unzip the file to get the `csv` file.

We then need to filter the high-quality buildings that correspond to the ROI:

```python
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

# Load the buildings using pandas (after downloading & unzipping the file [check Step 3])
df = pd.read_csv("./data/183_buildings.csv")

# Filter for high-confidence buildings
df = df[df["confidence"] > 0.83]

# Get the geometries of the buildings and load them into objects (from strings)
geometries = [loads(geom) for geom in df["geom"].tolist()]

# Load the original ROI
roi = gpd.read_file("./data/roi.geojson").geometry[0]

# Filter the geometries that are inside the region of interest
nairobi_buildings = [geom for geom in geometries if roi.contains(geom)]

# Mark the buildings as `1` and Save the file
data = {"y": [1]*len(nairobi_buildings), "geometry": nairobi_buildings}
gpd.GeoDataFrame(data=data, crs="EPSG:4326").to_file("./data/nairobi_buildings.geojson")
```

## Step 4: Create the `train` & `validation` ROI Polygons using QGIS

We create two polygons in `QGIS` to define the `train` & `validation` regions of interest:

1. Go to `Layer > Create Layer > New GeoPackage Layer...` and create a new layer.
2. Use the edit tool to draw a polygon for the `train` region of interest.
3. Repeat the same for the `validation` region of interest.
4. Save the layers.

## Step 5: Crop the tile to export `X_train.tif` & `X_val.tif`

Now we need to crop the original tile to the `train` & `validation` regions of interest and create the `X_train.tif` & `X_val.tif` files:

```Python
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
            
# Set the paths
intput_path = "./data/tile.tif"
xtrain_path = "./data/X_train.tif"
xval_path = "./data/X_val.tif"

# Load the train and validation regions of interest
train_roi = gpd.read_file("./data/train_roi.gpkg").geometry[0]
val_roi = gpd.read_file("./data/val_roi.gpkg").geometry[0]

# Crop the tile to the train region of interest
crop_tile_save(intput_path, train_roi, xtrain_path)

# Crop the tile to the validation region of interest
crop_tile_save(intput_path, val_roi, xval_path)
```

## Step 5: Rasterize the Open Buildings polygons to export `y_train.tif` & `y_val.tif`

We create a function to rasterize the Open Buildings polygons to create the `y_train.tif` & `y_val.tif` files:

```python
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
    
# Load the buildings
buildings = gpd.read_file("./data/nairobi_buildings.geojson").to_crs(rio.open("./data/X_train.tif").crs)

# Rasterize the buildings to create the y_train.tif file
rasterize_label_geoms("./data/X_train.tif", buildings, "./data/y_train.tif")

# Rasterize the buildings to create the y_val.tif file
rasterize_label_geoms("./data/X_val.tif", buildings, "./data/y_val.tif")
```
