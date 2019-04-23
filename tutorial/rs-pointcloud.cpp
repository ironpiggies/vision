#include <librealsense2/rs.hpp>
#include "example.hpp"

#include <algorithm>

void register_glfw_callbacks(window& app, glfw_state& app_state);

int main(int argc, char* argv[]) try
{
    window app(1280, 720, "RealSense Pointcloud");
    glfw_state app_state;
    register_glfw_callbacks(app, app_state);

    rs2::pointcloud pc;
    rs2::points points;

    rs2::pipeline pipe;
    pipe.start();

    while (app) {
        auto frames = pipe.wait_for_frames();
        auto color = frames.get_color_frame();

        if (!color)
            color = frames.get_infrared_frame();

        pc.map_to(color);
        auto depth = frames.get_depth_frame();

        points = pc.calculate(depth);

        app_state.tex.upload(color);

        draw_pointcloud(app.width(), app.height(), app_state, points);
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}