# Play video back
import os, random
import pyrealsense2 as rs
import cv2 as cv
import numpy as np


class Playback:
    def __init__(self, filename):
        config = rs.config()
        config.enable_device_from_file(filename)
        
        # Set alignment
        self.align = rs.align(rs.stream.color)

        # Start pipeline
        self.pipe = rs.pipeline()
        profile = self.pipe.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def play(self):
        while True:
            frames = self.pipe.wait_for_frames()
            # To Do: Check align
            frames = self.align.process(frames)

            # Get color image
            color = frames.get_color_frame()
            color_img = np.asarray(color.get_data())
            color_img = cv.resize(color_img, (640, 360))

            # Get depth image
            depth = frames.get_depth_frame()
            depth_img = np.asarray(depth.get_data())
            depth_img = cv.resize(depth_img, (640, 360))

            cv.imshow("color", cv.cvtColor(color_img, cv.COLOR_BGR2HSV))
            cv.imshow("depth", depth_img)

            # Update every wait_time milliseconds, and exit on ctrl-C
            k = cv.waitKey(100)
            if k == 27:
                break


directory = "videos/"
filename = directory + random.choice(os.listdir(directory))
playback = Playback(filename)
print("play from file: {}".format(filename))
playback.play()