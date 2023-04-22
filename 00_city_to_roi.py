"""Takes a city name from the user and saves the ROI of the city."""
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


def main(city_name, output_path):

    # Get the train & validation ROIs
    roi = city_bounding_box(city_name)

    # Save it to a GeoJSON file
    gpd.GeoDataFrame(geometry=[roi], crs="EPSG:4326").to_file(output_path)


if __name__ == "__main__":

    # Get the city name from the CLI arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Nairobi", help="The name of the city")
    parser.add_argument("--output_path", type=str, default="./data/roi.geojson", help="The path to save the ROI to")
    args = parser.parse_args()

    # Run the main function
    main(args.city_name, args.output_path)