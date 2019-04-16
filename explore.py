import pyrealsense2 as rs
import numpy as np


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def filter_zero_points(points):
    return points[~np.all(points == 0, axis=1)]


def filter_by_depth(points, zmax=0.95):
    return points[points[:, 2] < zmax]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

pipeline = rs.pipeline()
pipeline.start(config)


while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame: continue

    pc = rs.pointcloud()

    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)

    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

    points = filter_zero_points(verts)

    points = filter_by_depth(points, zmax=0.94)

    # def points_to_array(points):
    #     new_points = np.zeros(shape=(len(points), 3))
    #     for i, pt in enumerate(points):
    #         new_points[i] = list(pt)
    #     return new_points

    # array = points_to_array(points)
    #

    xs = points[::4, 0]
    ys = points[::4, 1]
    zs = points[::4, 2]

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 2])

    ax.scatter(xs, ys, zs, cmap="viridis", linewidth=0.5)

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zlow, zhigh)
    #     ax.scatter(xs, ys, zs, c=c, marker=m)

    plt.draw()
    plt.pause(1)
    ax.cla()



