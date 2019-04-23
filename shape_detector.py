import cv2
import numpy as np
from matplotlib import pyplot as plt


import pyrealsense2 as rs


config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

pipeline = rs.pipeline()
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()

    img = np.asanyarray(color.get_data())

    if img is not None:
        break

edges = cv2.Canny(img, 100, 200)

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()