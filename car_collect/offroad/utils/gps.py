import pyproj
import math

def determine_utm_zone(longitude):
    """Determine the UTM zone for a given longitude"""
    return int((longitude + 180) / 6) + 1

def lla_to_utm(lat, lon):
    """Convert LLA to UTM coordinates"""
    zone = determine_utm_zone(lon)
    utm_proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
    x, y = utm_proj(lon, lat)
    return x, y

def lla_to_xy(lat, lon, reference_lat=0, reference_lon=0):
    """
    Convert latitude and longitude to X and Y coordinates using the equirectangular approximation.
    Parameters:
    - lat, lon: Latitude and longitude of the point to convert.
    - reference_lat, reference_lon: Latitude and longitude of the reference point (defaults to 0,0).
    
    Returns:
    A list [x, y] representing the X and Y coordinates in meters.
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    reference_lat_rad = math.radians(reference_lat)
    reference_lon_rad = math.radians(reference_lon)
    
    # Equirectangular approximation
    x = (lon_rad - reference_lon_rad) * math.cos((reference_lat_rad + lat_rad) / 2)
    y = lat_rad - reference_lat_rad
    
    # Convert radians to meters
    x = x * R
    y = y * R
    
    return [x, y]