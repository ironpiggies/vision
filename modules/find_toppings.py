# Contains functions for finding toppings
from collections import Counter
import os
import random

import numpy as np
import pyrealsense2 as rs
import cv2 as cv


class HashableArray:
    def __init__(self, array):
        self.array = array

    def __hash__(self):
        return hash(tuple(self.array))


class HashableTopping:
    def __init__(self, data):
        self.data = data

    def __hash__(self):
        return hash(tuple(self.data[:2]))  # use (x, y) as hash


class CirclesCounter:
    """ Keep a running count of frequent circles
    """
    def __init__(self):
        self.max_keep = 20  # Max number of circles to keep
        self.cnt = Counter()

    def update(self, circles):
        for new_cir in circles:
            counted = False
            for hashable_old_cir in self.cnt:
                old_cir = hashable_old_cir.array
                if self.same(new_cir, old_cir):
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
        counts = []
        for c in mc:
            circles.append(c[0].array)
            counts.append(c[1])
        return circles, counts

    def clear(self):
        self.cnt = Counter()

    def same(self, a, b):
        """
        :param a: circle a (x, y, r)
        :param b: circle b (x, y, r)
        :return: True if a and b likely the same to some precision
        """
        ax, ay, ar = a
        bx, by, br = b
        tol = 5
        return abs(ax-bx) < tol and abs(ay-by) < tol and abs(ar-br) < tol


class ToppingCounter:
    """ Keep a running count of frequent toppings other than circles
    """
    def __init__(self):
        self.max_keep = 20  # Max number of circles to keep
        self.cnt = Counter()

    def update(self, toppings):
        for new_topping in toppings:
            counted = False
            for hashable_old_topping in self.cnt:
                old_topping = hashable_old_topping.data
                if self.same(new_topping, old_topping):
                    self.cnt[hashable_old_topping] += 1
                    counted = True
                    break
            if not counted:
                self.cnt[HashableTopping(new_topping)] += 1

        if len(self.cnt) > self.max_keep:
            mc = self.cnt.most_common(self.max_keep)
            self.cnt = Counter()
            for c in mc:
                self.cnt[c[0]] = c[1]

    def most_common(self, n=5):
        mc = self.cnt.most_common(n)
        toppings = []
        counts = []
        for c in mc:
            toppings.append(c[0].data)
            counts.append(c[1])
        return toppings, counts

    def clear(self):
        self.cnt = Counter()

    def same(self, a, b):
        """
        :param a: topping a (x, y, contour)
        :param b: topping b (x, y, contour)
        :return: True if a and b likely the same to some precision
        """
        ax, ay, _ = a
        bx, by, _ = b
        tol = 5
        return abs(ax-bx) < tol and abs(ay-by) < tol


class ChefVision:
    def __init__(self, filename=None, dev=False):
        # if we are in development phase
        self.dev = dev

        # Configuration
        config = rs.config()
        if filename:
            config.enable_device_from_file(filename)
        else:
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Set alignment
        self.align = rs.align(rs.stream.color)

        # Start pipeline
        self.pipe = rs.pipeline()
        profile = self.pipe.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def find_circles_by_HoughCircles(self, color_img):
        """
        :param color_img: color image read from RealSense
        :return: (a, b, c) where a are ingredient circles, b are pizza inner circles, c are pizza outer circle
        """
        blur = cv.GaussianBlur(color_img, (11, 11), 0)
        # gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(blur, 35, 70, apertureSize=3)

        circular_toppings = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 25, param1=100, param2=25, minRadius=20, maxRadius=32)
        if circular_toppings is not None:
            circular_toppings = np.round(circular_toppings[0, :]).astype("int").tolist()
        else:
            circular_toppings = []

        pizza_inners = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 25, param1=100, param2=25, minRadius=32,
                                       maxRadius=43)
        if pizza_inners is not None:
            pizza_inners = np.round(pizza_inners[0, :]).astype("int").tolist()
        else:
            pizza_inners = []

        pizza_outers = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 25, param1=100, param2=25, minRadius=200,
                                       maxRadius=215)
        if pizza_outers is not None:
            pizza_outers = np.round(pizza_outers[0, :]).astype("int").tolist()
        else:
            pizza_outers = []

        return circular_toppings, pizza_inners, pizza_outers

    def filter_toppings_by_depth(self, toppings, depth_img, minmax=(0.5, 0.8)):
        """
        :param toppings: list of toppings, each being (x, y, something)
        :param depth_img: depth image
        :param minmax: min and max thresholds
        :return:
        """
        min, max = minmax
        new_toppings = []
        for topping in toppings:
            x, y, _ = topping
            if min < self.get_depth(depth_img, x, y) < max:
                new_toppings.append(topping)
        return new_toppings

    def filter_pizza(self, pizza_inner, pizza_outer):
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
                if not pushed and self.circle_inside(inner, outer):
                    new_inner.append(inner)
                    pushed = True
            for i, outer in enumerate(pizza_outer_copy):
                if np.array_equal(outer, np.zeros_like(outer)):
                    continue
                inside = self.circle_inside(inner, outer)
                if inside:
                    new_outer.append(np.copy(outer))
                    pizza_outer_copy[i] = [0, 0, 0]
                    if not pushed:
                        new_inner.append(inner)
                        pushed = True
        return new_inner, new_outer

    def circle_inside(self, a, b):
        """
        :param a: a circle (x, y, r)
        :param b: a circle (x, y, r)
        :return: True if a is inside b
        """
        ax, ay, ar = a
        bx, by, br = b
        center_dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        if center_dist + ar <= br:
            return True
        return False

    def get_depth(self, depth_img, x ,y):
        """
        :param depth_img: depth image of shape (h, w)
        :param x: x index
        :param y: y index
        :return: depth at (x, y) in meter
        """
        return depth_img[y, x] * self.depth_scale

    def filter_red_circle(self, circles, color_img):
        """
        :param circles: list of circles that represent potential red circles
        :param color_img: color image read from RealSense
        :return: circles that match red color
        """
        red_cirs = []
        black_rings = []
        hsv_img = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
        for circle in circles:
            x, y, r = circle
            if self.is_red_color(hsv_img[y, x]):
                red_cirs.append(circle)
            else:
                black_rings.append(circle)
        return red_cirs, black_rings

    def is_red_color(self, pixel):
        """
        :param pixel: color pixel in HSV
        :return: True if red
        """
        h, s, v = pixel
        if 0 <= h <= 10 or 160 <= h <= 180:
            return True
        return False

    def get_confident_ones(self, stuffs, stuff_counts, threshold):
        """
        :param stuffs: list of stuff coordinates
        :param stuff_counts: corresponding list of stuff counts
        :param threshold: min counts to regard as confident
        :return: list of confident stuff
        """
        confident_ones = []
        for stuff, stuff_count in zip(stuffs, stuff_counts):
            if stuff_count > threshold:
                confident_ones.append(stuff)
        return confident_ones

    def get_xyz(self, center, depth_frame, depth_img):
        """
        :param center: (x, y) of a topping
        :param depth_frame: depth frame from RealSense
        :param depth_img: depth image of shape (h, w)
        :return: (x, y, z) of its center
        """
        h_im, w_im = depth_img.shape
        w_frame = depth_frame.get_width()
        h_frame = depth_frame.get_height()
        x, y = center
        depth_pixel = [int(float(x) / w_im * w_frame), int(float(y) / h_im * h_frame)]
        depth = depth_frame.get_distance(depth_pixel[0], depth_pixel[1])
        depth_intr = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intr, depth_pixel, depth)
        return depth_point

    def find_yellow_triangles(self, color_img):
        """
        :param color_img: color image read from RealSense
        :return: yellow_triangles: list of (x_center, y_center, contours)
        """
        # HSV masking
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
        hsv_image = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
        masked_hsv = cv.inRange(hsv_image, lower, upper)

        # Find connected components
        retval, labels = cv.connectedComponents(masked_hsv)

        yellow_triangles = []
        for label in range(retval):
            new_img = (labels == label).astype("int")
            # size thresholding
            if 500 < np.sum(new_img) < 100000:
                new_img = np.array(new_img*255, dtype=np.uint8)
                _, thresh = cv.threshold(new_img, 127, 255, cv.THRESH_BINARY)
                moments = cv.moments(thresh)
                x_center = int(moments["m10"] / moments["m00"])
                y_center = int(moments["m01"] / moments["m00"])
                contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                yellow_triangles.append([x_center, y_center, contours])

        return yellow_triangles

    def find_blue_fishes(self, color_img):
        """
        :param color_img: color image read from RealSense
        :return: blue_fishes: list of (x_center, y_center, contours)
        """
        # HSV masking
        lower = np.array([100, 150, 0])
        upper = np.array([140, 255, 255])
        hsv_image = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
        masked_hsv = cv.inRange(hsv_image, lower, upper)

        # Find connected components
        retval, labels = cv.connectedComponents(masked_hsv)

        blue_fishes = []
        for label in range(retval):
            new_img = (labels == label).astype("int")
            # size thresholding
            if 150 < np.sum(new_img) < 800:
                new_img = np.array(new_img * 255, dtype=np.uint8)
                _, thresh = cv.threshold(new_img, 127, 255, cv.THRESH_BINARY)
                moments = cv.moments(thresh)
                x_center = int(moments["m10"] / moments["m00"])
                y_center = int(moments["m01"] / moments["m00"])
                contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                blue_fishes.append([x_center, y_center, contours])

        return blue_fishes

    def find_pink_squares(self, color_img):
        """
        :param color_img: color image read from RealSense
        :return: pink_squares: list of (x_center, y_center, contours)
        """
        # HSV masking
        lower = np.array([150, 100, 110])
        upper = np.array([221, 255, 255])
        hsv_image = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
        masked_hsv = cv.inRange(hsv_image, lower, upper)

        # Find connected components
        retval, labels = cv.connectedComponents(masked_hsv)

        pink_squares = []
        for label in range(retval):
            new_img = (labels == label).astype("int")
            # size thresholding
            if 1000 < np.sum(new_img) < 3000:
                new_img = np.array(new_img * 255, dtype=np.uint8)
                _, thresh = cv.threshold(new_img, 127, 255, cv.THRESH_BINARY)
                moments = cv.moments(thresh)
                x_center = int(moments["m10"] / moments["m00"])
                y_center = int(moments["m01"] / moments["m00"])
                contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                pink_squares.append([x_center, y_center, contours])

        return pink_squares

    def find_toppings(self, _3d_coord=True):
        """
        :param _3d_coord: whether return toppings as 3D positions or 2D pixels w/ radius
        :return: dictionary with topping name as key, and list of coords as values, (if 2D, color image is also returned)
        """
        red_cir_counter = CirclesCounter()
        black_ring_counter = CirclesCounter()
        yellow_triangle_counter = ToppingCounter()
        blue_fish_counter = ToppingCounter()
        pink_square_counter = ToppingCounter()
        pizza_inner_counter = CirclesCounter()
        pizza_outer_counter = CirclesCounter()

        total_time = 1.0  # seconds
        time_iterator = 0
        wait_time = 100  # milliseconds
        while True:
            time_iterator += 1
            frames = self.pipe.wait_for_frames()
            # To Do: Check align
            frames = self.align.process(frames)

            # Get color image
            color = frames.get_color_frame()
            color_img = np.asarray(color.get_data())
            color_img = cv.resize(color_img, (1920, 1080))

            # Get depth image
            depth = frames.get_depth_frame()
            depth_img = np.asarray(depth.get_data())
            depth_img = cv.resize(depth_img, (1920, 1080))

            # Hough Circles combined with size filtering
            circular_toppings, pizza_inners, pizza_outers = self.find_circles_by_HoughCircles(color_img)

            # Depth filtering
            table_min, table_max = 0.5, 0.8
            circular_toppings = self.filter_toppings_by_depth(circular_toppings, depth_img, (table_min, table_max))
            pizza_inners = self.filter_toppings_by_depth(pizza_inners, depth_img, (table_min, table_max))
            pizza_outers = self.filter_toppings_by_depth(pizza_outers, depth_img, (table_min, table_max))

            # Color filter for red circles
            red_cirs, black_rings = self.filter_red_circle(circular_toppings, color_img)

            # Pizza-inner-must-be-inside-pizza-outer filter
            pizza_inners, pizza_outers = self.filter_pizza(pizza_inners, pizza_outers)

            # Find yellow triangles
            yellow_triangles = self.find_yellow_triangles(color_img)
            yellow_triangles = self.filter_toppings_by_depth(yellow_triangles, depth_img, (table_min, table_max))

            # Find blue fishes
            blue_fishes = self.find_blue_fishes(color_img)
            blue_fishes = self.filter_toppings_by_depth(blue_fishes, depth_img, (table_min, table_max))

            # Find pink squares
            pink_squares = self.find_pink_squares(color_img)
            pink_squares = self.filter_toppings_by_depth(pink_squares, depth_img, (table_min, table_max))

            # Update and get most common
            red_cir_counter.update(red_cirs)
            black_ring_counter.update(black_rings)
            yellow_triangle_counter.update(yellow_triangles)
            blue_fish_counter.update(blue_fishes)
            pink_square_counter.update(pink_squares)
            pizza_inner_counter.update(pizza_inners)
            pizza_outer_counter.update(pizza_outers)

            if self.dev:
                red_cirs, red_cir_counts = red_cir_counter.most_common(5)
                black_rings, black_ring_counts = black_ring_counter.most_common(5)
                yellow_triangles, yellow_triangle_counts = yellow_triangle_counter.most_common(5)
                blue_fishes, blue_fish_counts = blue_fish_counter.most_common(5)
                pink_squares, pink_square_counts = pink_square_counter.most_common(5)
                pizza_inners, pizza_inner_counts = pizza_inner_counter.most_common(9)
                pizza_outers, pizza_outer_counts = pizza_outer_counter.most_common(1)

                # Display detection
                toppings = {
                    "red_cirs": red_cirs,
                    "black_rings": black_rings,
                    "yellow_triangles": yellow_triangles,
                    "blue_fishes": blue_fishes,
                    "pink_squares": pink_squares,
                    "pizza_inners": pizza_inners,
                    "pizza_outers": pizza_outers
                }
                color_img_copy = draw_toppings(toppings, color_img)
                color_img_copy = cv.resize(color_img_copy, (640, 360))
                cv.imshow("detection", color_img_copy)

                # Update every wait_time milliseconds, and exit on ctrl-C
                k = cv.waitKey(wait_time)
                if k == 27:
                    break

                if wait_time * time_iterator > total_time * 1000:
                    time_iterator = 0
                    red_cir_counter.clear()
                    black_ring_counter.clear()
                    yellow_triangle_counter.clear()
                    blue_fish_counter.clear()
                    pink_square_counter.clear()
                    pizza_inner_counter.clear()
                    pizza_outer_counter.clear()

            else:
                if wait_time * time_iterator > total_time * 1000:
                    break

        red_cirs, red_cir_counts = red_cir_counter.most_common(5)
        black_rings, black_ring_counts = black_ring_counter.most_common(5)
        yellow_triangles, yellow_triangle_counts = yellow_triangle_counter.most_common(5)
        blue_fishes, blue_fish_counts = blue_fish_counter.most_common(5)
        pink_squares, pink_square_counts = pink_square_counter.most_common(5)
        pizza_inners, pizza_inner_counts = pizza_inner_counter.most_common(9)
        pizza_outers, pizza_outer_counts = pizza_outer_counter.most_common(1)

        # Get the ones that appear often enough
        count_threshold = 0*total_time*1000.0/wait_time
        confident_red_cirs = self.get_confident_ones(red_cirs, red_cir_counts, count_threshold)
        confident_black_rings = self.get_confident_ones(black_rings, black_ring_counts, count_threshold)
        confident_yellow_triangles = self.get_confident_ones(yellow_triangles, yellow_triangle_counts, count_threshold)
        confident_blue_fishes = self.get_confident_ones(blue_fishes, blue_fish_counts, count_threshold)
        confident_pink_squares = self.get_confident_ones(pink_squares, pink_square_counts, count_threshold)
        confident_pizza_inners = self.get_confident_ones(pizza_inners, pizza_inner_counts, count_threshold)
        confident_pizza_outers = self.get_confident_ones(pizza_outers, pizza_outer_counts, count_threshold)

        if not _3d_coord:
            return {
                "red_cirs": confident_red_cirs,
                "black_rings": confident_black_rings,
                "yellow_triangles": confident_yellow_triangles,
                "blue_fishes": confident_blue_fishes,
                "pink_squares": confident_pink_squares,
                "pizza_inners": confident_pizza_inners,
                "pizza_outers": confident_pizza_outers
            }, color_img
        else:
            red_cir_coords = [self.get_xyz(circle[:2], depth_frame=depth, depth_img=depth_img) for circle in confident_red_cirs]
            black_ring_coords = [self.get_xyz(circle[:2], depth_frame=depth, depth_img=depth_img) for circle in confident_black_rings]
            yellow_triangle_coords = [self.get_xyz(triangle[:2], depth_frame=depth, depth_img=depth_img) for triangle in confident_yellow_triangles]
            blue_fish_coords = [self.get_xyz(fish[:2], depth_frame=depth, depth_img=depth_img) for fish in confident_blue_fishes]
            pink_square_coords = [self.get_xyz(square[:2], depth_frame=depth, depth_img=depth_img) for square in confident_pink_squares]
            pizza_inner_coords = [self.get_xyz(circle[:2], depth_frame=depth, depth_img=depth_img) for circle in confident_pizza_inners]
            pizza_outer_coords = [self.get_xyz(circle[:2], depth_frame=depth, depth_img=depth_img) for circle in confident_pizza_outers]
            return {
                "red_cirs": red_cir_coords,
                "black_rings": black_ring_coords,
                "yellow_triangles": yellow_triangle_coords,
                "blue_fishes": blue_fish_coords,
                "pink_squares": pink_square_coords,
                "pizza_inners": pizza_inner_coords,
                "pizza_outers": pizza_outer_coords
            }


def draw_toppings(toppings, color_img):
    red_cirs, black_rings, pizza_inners, pizza_outers = toppings["red_cirs"], toppings["black_rings"], \
                                                        toppings["pizza_inners"], toppings["pizza_outers"]
    yellow_triangles = toppings["yellow_triangles"]
    blue_fishes = toppings["blue_fishes"]
    pink_squares = toppings["pink_squares"]

    color_img_copy = np.copy(color_img)
    for (x, y, r) in red_cirs:
        cv.circle(color_img_copy, (x, y), r, (0, 255, 0), 4)
    for (x, y, r) in black_rings:
        cv.circle(color_img_copy, (x, y), r, (255, 0, 255), 4)
    for (x, y, r) in pizza_inners:
        cv.circle(color_img_copy, (x, y), r, (255, 0, 0), 4)
    for (x, y, r) in pizza_outers:
        cv.circle(color_img_copy, (x, y), r, (0, 0, 255), 4)

    for (x, y, contours) in yellow_triangles:
        cv.drawContours(color_img_copy, contours, -1, [255, 255, 0], 4)
    for (x, y, contours) in blue_fishes:
        cv.drawContours(color_img_copy, contours, -1, [0, 255, 255], 4)
    for (x, y, contours) in pink_squares:
        cv.drawContours(color_img_copy, contours, -1, [0, 0, 0], 4)

    return color_img_copy


# Copied from Nic's vision_helper.py
def apply_hsv_mask(img, lower_mask, upper_mask):
    #might have to change this up to make it most useful for what type img, lower_mask etc is
    '''
    lower_mask and upper_mask are lists of 3 values representing h,s,v
    img is np array of bgr image (default opencv type)
    returns masked version of img as bgr
    '''
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    masked_img = np.zeros(shape=hsv_img.shape[:2])
    for lower, upper in zip(lower_mask, upper_mask):
        lower_mask_array = np.array(lower)
        upper_mask_array = np.array(upper)
        masked_img += cv.inRange(hsv_img, lower_mask_array, upper_mask_array)

    return masked_img


# Largely copied from Nic's color_code.py
def denoise(img):
    # erode first then dilate to reduce noise
    size = 3
    iters = 1

    kernel = np.ones((size, size), np.uint8)
    erosion = cv.erode(img, kernel, iterations=iters)
    dilation = cv.dilate(erosion, kernel, iterations=iters+2)
    return dilation


def print_toppings_dict(toppings):
    """ Pretty print with 3 decimal places
    """
    for k, v_list in toppings.items():
        print("{}:\n {}".format(k, np.round(v_list, 3)))
    print("================")


def main():
    """ Code for testing functionalities
    """
    USE_RECORDING = False
    if USE_RECORDING:  # Use recording
        directory = "videos/"
        filename = directory + random.choice(os.listdir(directory))
        print("play from file: {}".format(filename))
        chef_vision = ChefVision(filename=filename)
    else:
        print("use current video")
        chef_vision = ChefVision()

    while True:
        toppings, color_img = chef_vision.find_toppings(_3d_coord=False)

        # Display detection
        color_img_copy = draw_toppings(toppings, color_img)

        color_img_copy = cv.resize(color_img_copy, (640, 360))
        cv.imshow("detection", color_img_copy)

        # Update every wait_time milliseconds, and exit on ctrl-C
        k = cv.waitKey(100)
        if k == 27:
            break


def test_continuous():
    """ Test for displaying detection results continuously - mainly for development
    """
    USE_RECORDING = True
    if USE_RECORDING:  # Use recording
        directory = "videos/"
        filename = directory + random.choice(os.listdir(directory))
        print("play from file: {}".format(filename))
        chef_vision = ChefVision(filename=filename, dev=True)
    else:
        print("use real video")
        chef_vision = ChefVision(dev=True)

    chef_vision.find_toppings()


if __name__ == "__main__":
    main()
    # test_continuous()