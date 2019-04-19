#include <librealsense2/rs.hpp>
#include <iostream>


int main(int argc, char * argv[]) try
{

    rs2::pipeline p;

    p.start();

    while (true) {
        rs2::frameset frames = p.wait_for_frames();

        rs2::depth_frame depth = frames.get_depth_frame();

        float width = depth.get_width();
        float height = depth.get_height();

        float dist_to_center = depth.get_distance(width/2, height/2);
        std::cout << "The camera is facing an object " << dist_to_center << " meters away\n";
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}