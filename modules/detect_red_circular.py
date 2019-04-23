from collections import Counter

import pyrealsense2 as rs
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt


# Configuration
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Set alignment
align = rs.align(rs.stream.depth)

# Start pipeline
pipe = rs.pipeline()
profile = pipe.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()


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


def find_circles_by_HoughCircles(color_img):
    """
    :param color_img: color image read from RealSense
    :return: (a, b, c) where a are ingredient circles, b are pizza inner circles, c are pizza outer circle
    """
    blur = cv.GaussianBlur(color_img, (7, 7), 0)
    # gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(blur, 50, 150, apertureSize=3)
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 6, param1=40, param2=25, minRadius=0, maxRadius=70)

    ingredients = []
    pizza_inner = []
    pizza_outer = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # convert circles to appropriate form
        for circle in circles:
            x, y, r = circle
            if r < 10:
                ingredients.append(circle)
            if r > 10 and r < 14:
                pizza_inner.append(circle)
            if r > 64 and r < 70:
                pizza_outer.append(circle)

    return ingredients, pizza_inner, pizza_outer


def filter_red_circle(circles, color_img):
    """
    :param circles: list of circles that represent potential red circles
    :param color_img: color image read from RealSense
    :return: circles that match red color
    """
    new_cirs = []
    hsv_img = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
    for circle in circles:
        x, y, r = circle
        if is_red_color(hsv_img[y, x]):
            new_cirs.append(circle)
    return new_cirs


def is_red_color(pixel):
    """
    :param pixel: color pixel in HSV
    :return: True if red
    """
    h, s, v = pixel
    if 0 <= h <= 10 or 160 <= h <= 180:
        return True
    return False


def filter_circles_by_depth(circles, minmax=(0.5, 0.8)):
    min, max = minmax
    new_cirs = []
    for cir in circles:
        x, y, _ = cir
        if min < get_depth(depth_img, x, y) < max:
            new_cirs.append(cir)
    return new_cirs


def filter_pizza(pizza_inner, pizza_outer):
    """
    :param pizza_inner: list of circles that represent potential pizza inner circles
    :param pizza_outer: list of circles that represent potential pizza outer circles
    :return: pizza_inner, pizza_outer, where a pizza inner circle must be inside a pizza outer, and vice versa
    """
    pizza_outer_copy = np.copy(pizza_outer)
    new_inner = []
    new_outer = []
    for inner in pizza_inner:
        pushed = False  # Keep track if inner is appended
        for outer in new_outer:
            if not pushed and circle_inside(inner, outer):
                new_inner.append(inner)
                pushed = True
        for i, outer in enumerate(pizza_outer_copy):
            if np.array_equal(outer, np.zeros_like(outer)):
                continue
            inside = circle_inside(inner, outer)
            if inside:
                new_outer.append(np.copy(outer))
                pizza_outer_copy[i] = [0, 0, 0]
                if not pushed:
                    new_inner.append(inner)
                    pushed = True
    return new_inner, new_outer


def circle_inside(a, b):
    """
    :param a: a circle (x, y, r)
    :param b: a circle (x, y, r)
    :return: True if a is inside b
    """
    ax, ay, ar = a
    bx, by, br = b
    center_dist = np.sqrt((ax-bx)**2 + (ay-by)**2)
    if center_dist + ar <= br:
        return True
    return False


def filter_by_depth(color_img, depth_img, minmax=(0, 1)):
    """
    :param color_img: color image of shape (h, w, 3)
    :param depth_img: depth image of shape (h, w)
    :param minmax: tuple of min and max threshold
    :return: color image where pixel is set to black for depth outside range
    """
    assert color_img.shape[:2] == depth_img.shape
    h, w = depth_img.shape
    min, max = minmax
    color_img_copy = np.copy(color_img)
    depth_img_copy = convert_depth_scale(depth_img, depth_scale)
    for i in range(h):
        for j in range(w):
            if depth_img_copy[i, j] >= max or depth_img_copy[i, j] <= min:
                color_img_copy[i, j] = [0, 0, 0]
    return color_img_copy


def convert_depth_scale(depth_img, depth_scale):
    """
    :param depth_img: depth image of shape (h, w)
    :param depth_scale: depth scale as a float
    :return: depth image transformed to meters scale
    """
    depth_img_copy = np.copy(depth_img) * depth_scale
    return depth_img_copy


def get_depth(depth_img, x ,y):
    """
    :param depth_img: depth image of shape (h, w)
    :param x: x index
    :param y: y index
    :return: depth at (x, y) in meter
    """
    return depth_img[y, x] * depth_scale


class HashableArray:
    def __init__(self, array):
        self.array = array

    def __hash__(self):
        return hash(tuple(self.array))


class CirclesCounter:
    """ Keep a running count of frequent circles
    """
    def __init__(self):
        self.max_keep = 100  # Max number of circles to keep
        self.cnt = Counter()

    def update(self, circles):
        for new_cir in circles:
            counted = False
            for hashable_old_cir in self.cnt:
                old_cir = hashable_old_cir.array
                if same_circle(new_cir, old_cir):
                    self.cnt[hashable_old_cir] += 1
                    counted = True
                    break
            if not counted:
                self.cnt[HashableArray(new_cir)] += 1

        if len(self.cnt) > self.max_keep:
            mc = self.cnt.most_common(self.max_keep)
            self.cnt = Counter()
            for c in mc:
                self.cnt[c[0]] = c[1]

    def most_common(self, n=5):
        mc = self.cnt.most_common(n)
        circles = []
        for c in mc:
            circles.append(c[0].array)
        return circles

    def clear(self):
        self.cnt = Counter()


def same_circle(a, b):
    """
    :param a: circle a (x, y, r)
    :param b: circle b (x, y, r)
    :return: True if a and b likely the same to some precision
    """
    ax, ay, ar = a
    bx, by, br = b
    tol = 5
    return abs(ax-bx) < tol and abs(ay-by) < tol and abs(ar-br) < tol


red_cir_counter = CirclesCounter()
pizza_inner_counter = CirclesCounter()
pizza_outer_counter = CirclesCounter()


clear_time = 10.0  # number of seconds between clearing counters
time_it = 0
wait_time = 100  # in milliseconds, time to wait in each loop
while True:
    time_it += 1
    frames = pipe.wait_for_frames()
    # frames = align.process(frames)

    color = frames.get_color_frame()

    color_img = np.asarray(color.get_data())
    color_img = cv.resize(color_img, (640, 360))  # resize to size tuned for circular detection

    depth = frames.get_depth_frame()
    depth_img = np.asarray(depth.get_data())
    depth_img = cv.resize(depth_img, (640, 360))

    # color_img = filter_by_depth(color_img, depth_img, minmax=(0.5, 0.85))

    # cv.imshow('color', color_img)
    blur = cv.GaussianBlur(color_img, (7, 7), 0)
    # gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(blur, 50, 150, apertureSize=3)
    edges_3_channel = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    # orig.set_data(color_img)

    ingredients, pizza_inner, pizza_outer = find_circles_by_HoughCircles(color_img)

    table_min, table_max = 0.5, 0.8
    red_cir = filter_circles_by_depth(ingredients, (table_min, table_max))
    pizza_inner = filter_circles_by_depth(pizza_inner, (table_min, table_max))
    pizza_outer = filter_circles_by_depth(pizza_outer, (table_min, table_max))

    red_cir = filter_red_circle(red_cir, color_img)
    pizza_inner, pizza_outer = filter_pizza(pizza_inner, pizza_outer)

    # Update and get most common
    red_cir_counter.update(red_cir)
    pizza_inner_counter.update(pizza_inner)
    pizza_outer_counter.update(pizza_outer)

    red_cir = red_cir_counter.most_common(5)
    pizza_inner = pizza_inner_counter.most_common(9)
    pizza_outer = pizza_outer_counter.most_common(1)

    # print(pizza_outer)

    for (x, y, r) in red_cir:
        cv.circle(color_img, (x, y), r, (0, 255, 0), 4)
    for (x, y, r) in pizza_inner:
        cv.circle(color_img, (x, y), r, (255, 0, 0), 4)
    for (x, y, r) in pizza_outer:
        cv.circle(color_img, (x, y), r, (0, 0, 255), 4)


    # cv.imshow('edges', edges)
    # detect.set_data(edges)

    combined = np.vstack((color_img, edges_3_channel))
    cv.imshow("combined", combined)

    # fig.canvas.draw()

    # Update every wait_time milliseconds, and exit on ctrl-C
    k = cv.waitKey(wait_time)
    if k == 27:
        break

    if wait_time*time_it > clear_time*1000:
        time_it = 0
        red_cir_counter.clear()
        pizza_inner_counter.clear()
        pizza_outer_counter.clear()