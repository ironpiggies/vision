default: hello-world

hello-world: hello-world.cpp
	g++ -std=c++11 $< -o $@ -lrealsense2

rs-capture: rs-capture.cpp
	g++ -std=c++11 $< -o $@ -lrealsense2 -lglfw -lGL -lGLU

clean:
	-rm -f hello-world rs-capture

run: hello-world
	./hello-world
