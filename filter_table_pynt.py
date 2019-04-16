import pyrealsense2 as rs
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd

def point_to_pcl(point):
    pcl_point = pcl.PointCloud()
    sp = point.get_profile().as_video_stream_profile()
    pcl_point.width = sp.width()
    pcl_point.height = sp.height()
    pcl_point.is_dense = false
    pcl_point.resize(point.size())
    ptr = point.get_vertices()
    for p in range(0, pcl_point.points.size()):
        p.x = ptr.x
        p.y = ptr.y
        p.z = ptr.z
        ptr = ptr + 1

    return pcl_point


pipeline = rs.pipeline()
pipeline.start()

while True:
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth: continue

    pc = rs.pointcloud()
    points = pc.calculate(depth)

    if points:
        break

vtx = pd.DataFrame(np.asanyarray(points.get_vertices()), columns=["x", "y", "z"])
print(vtx)
cloud = PyntCloud(vtx)
print(cloud)

# cloud = point_to_pcl(points)
# visual = pcl.pcl_visualization.CloudViewing()
#
# visual.ShowColorCloud(cloud, b'cloud')
#
# flag = True
# while flag:
#     flag != visual.WasStopped()

