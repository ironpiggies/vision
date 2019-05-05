# Record video
import pyrealsense2 as rs
import datetime
import time


class Recorder:
    def __init__(self):
        # Configuration
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%y-%m-%d_%H_%M_%S")
        filename = "videos/{}.bag".format(current_time)
        print("save to file: {}".format(filename))
        self.config.enable_record_to_file(filename)

        # Start pipeline
        self.pipe = rs.pipeline()

    def record(self):
        self.pipe.start(self.config)
        time.sleep(20)
        self.pipe.stop()


recorder = Recorder()
recorder.record()
