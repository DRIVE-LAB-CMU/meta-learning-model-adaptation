import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pyproj
DATA_DIR = "../data"

with open(os.path.join(DATA_DIR, 'data-20231216-185104.json'), "r") as f:
    data = json.load(f)

locations = []
for st in data['state']:
    locations.append(st['loc'])
locations = np.array(locations)
# print(locations[0])
# locations[:] = locations[:] - locations[0]
# plt.plot(locations[:, 0], locations[:, 1])
# plt.show()


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
gps_locations = locations[:, :2]

# Taking the first location as the reference point
reference_point_utm = lla_to_utm(gps_locations[0][0], gps_locations[0][1])
utm_coords = []

for lat, lon in gps_locations:
    if np.isnan(lon):
        continue
    # print(lon)
    x, y = lla_to_utm(lat, lon)
    utm_coords.append([x, y])

# Subtracting the reference point to get relative coordinates
utm_coords = np.array(utm_coords) - reference_point_utm

plt.plot(utm_coords[:, 0], utm_coords[:, 1])
plt.show()

