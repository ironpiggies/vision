default: hello-world

hello-world: hello-world.cpp
	g++ -std=c++11 $< -o $@ -lrealsense2

rs-capture: rs-capture.cpp
	g++ -std=c++11 $< -o $@ -lrealsense2 -lglfw -lGL -lGLU

rs-pointcloud: rs-pointcloud.cpp
	g++ -std=c++11 $< -o $@ -lrealsense2 -lglfw -lGL -lGLU

rs-imshow: rs-imshow.cpp
	g++ -std=c++11 $< -o $@ -lrealsense2 `pkg-config --cflags --libs opencv`

rs-align: rs-align.cpp ../third-party/imgui/imgui.cpp ../third-party/imgui/imgui_draw.cpp ../third-party/imgui/imgui_impl_glfw.cpp
	g++ -std=c++11 $^ -o $@ -lrealsense2 -lglfw -lGL -lGLU -I../third-party/imgui/

rs-align-advanced: rs-align-advanced.cpp ../third-party/imgui/imgui.cpp ../third-party/imgui/imgui_draw.cpp ../third-party/imgui/imgui_impl_glfw.cpp
	g++ -std=c++11 $^ -o $@ -lrealsense2 -lglfw -lGL -lGLU -I../third-party/imgui/

rs-dnn: rs-dnn.cpp
	g++ -std=c++11 $< -o $@ -lrealsense2 `pkg-config --cflags --libs opencv`

clean:
	-rm -f hello-world rs-capture rs-pointcloud rs-imshow rs-align

run: hello-world
	./hello-world
