import pyrealsense2 as rs
import pcl-1.7.2 as pcl

import pcl.pcl_visualization


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

cloud = point_to_pcl(points)
visual = pcl.pcl_visualization.CloudViewing()

visual.ShowColorCloud(cloud, b'cloud')

flag = True
while flag:
    flag != visual.WasStopped()

