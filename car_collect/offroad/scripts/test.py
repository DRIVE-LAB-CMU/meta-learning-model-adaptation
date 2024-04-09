import numpy as np
import pyproj

def determine_utm_zone(longitude):
    """Determine the UTM zone for a given longitude"""
    return int((longitude + 180) / 6) + 1

def lla_to_utm(lat, lon):
    """Convert LLA to UTM coordinates"""
    zone = determine_utm_zone(lon)
    utm_proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
    x, y = utm_proj(lon, lat)
    return x, y

# Example usage
gps_locations = np.array([
    [40.44381633637052, -79.94555335498798], # Your GPS locations (latitude, longitude)
    # Add more points as needed
])

# Taking the first location as the reference point
reference_point_utm = lla_to_utm(gps_locations[0][0], gps_locations[0][1])
utm_coords = []

for lat, lon in gps_locations:
    x, y = lla_to_utm(lat, lon)
    utm_coords.append([x, y])

# Subtracting the reference point to get relative coordinates
utm_coords = np.array(utm_coords) - reference_point_utm

print(utm_coords)
