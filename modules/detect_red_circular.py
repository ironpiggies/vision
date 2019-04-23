import pyrealsense2 as rs
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt


config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 800, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

pipe = rs.pipeline()
pipe.start(config)

# plt.ion()
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# orig = ax1.imshow(np.zeros((1080, 1920, 3)), cmap='gray')
# # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#
# ax2 = fig.add_subplot(122)
# detect = ax2.imshow(np.zeros((1080, 1920)), cmap='gray')
# # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# fig.canvas.draw()


def find_circles(color_img):
    """
    :param color_img: color image read from RealSense
    :return: (a, b, c) where a are ingredient circles, b are pizza inner circles, c are pizza outer circle
    """
    blur = cv.GaussianBlur(color_img, (7, 7), 0)
    # gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(blur, 50, 150, apertureSize=3)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 5, param1=40, param2=25, minRadius=0, maxRadius=70)
    circles = np.round(circles[0, :]).astype("int")  # convert circles to appropriate form

    ingredients = []
    pizza_inner = []
    pizza_outer = []
    for circle in circles:
        x, y, r = circle
        if r < 10:
            ingredients.append(circle)
        if r > 10 and r < 14:
            pizza_inner.append(circle)
        if r > 65 and r < 68:
            pizza_outer.append(circle)

    return ingredients, pizza_inner, pizza_outer


while True:
    frames = pipe.wait_for_frames()

    color = frames.get_color_frame()

    color_img = np.asarray(color.get_data())
    color_img = cv.resize(color_img, (1920/3, 1080/3))
    # cv.imshow('color', color_img)

    # orig.set_data(color_img)

    edges = cv.Canny(color_img, 100, 200)

    ingredients, pizza_inner, pizza_outer = find_circles(color_img)
    for (x, y, r) in ingredients:
        cv.circle(color_img, (x, y), r, (0, 255, 0), 4)
    for (x, y, r) in pizza_inner:
        cv.circle(color_img, (x, y), r, (255, 0, 0), 4)
    for (x, y, r) in pizza_outer:
        cv.circle(color_img, (x, y), r, (0, 0, 255), 4)
    # circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 20, param1=100, param2=80, minRadius=0, maxRadius=0)
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #
    #     for (x, y, r) in circles:
    #         cv.circle(color_img, (x, y), r, (0, 255, 0), 4)
    #         # cv.rectangle(color_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    edges_3_channel = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    # cv.imshow('edges', edges)
    # detect.set_data(edges)

    combined = np.vstack((color_img, edges_3_channel))
    cv.imshow("combined", combined)

    # fig.canvas.draw()

    # Update every 10 milliseconds, and exit on ctrl-C
    k = cv.waitKey(100)
    if k == 27:
        break