import cv2
import json
import os 
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../data"
file_name = "data-20231218-153802"

with open(os.path.join(DATA_DIR, file_name), "r") as f:
    data = json.load(f)

data = data['left'][2]
np_arr = np.array(data)
# image_np = cv2.imdecode(data, cv2.IMREAD_COLOR)

# import pdb; pdb.set_trace()

# Example width and height of your image
width = 336
height = 188

# Reshape the array to the original image dimensions
# Assuming depth data is in np.uint16 format
image = np_arr.reshape((height, width, -1))
rgb_image = image[:, :, :3]
depth_image = image[:, :, 3]
print(depth_image)
assert np.all(depth_image == 255)
plt.imshow(depth_image, cmap='gray')
# plt.imshow(rgb_image)
plt.axis("off")
plt.show()
# Normalize the image for visualization
# Convert to a floating point image and scale to 0-1
# depth_normalized = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# Convert to 8-bit grayscale (0-255)
# depth_8bit = np.uint8(depth_normalized * 255)

# Display the image (optional)
# cv2.imshow('Depth Image', depth_8bit)