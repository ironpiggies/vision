#include <librealsense2/rs.hpp>
#include "example.hpp"
#include <imgui.h>
#include "imgui_impl_glfw.h"


enum class direction
{
    to_depth,
    to_color
};

void render_slider(rect location, float* alpha, direction* dir);

int main(int argc, char * argv[]) try {
    window app(1280, 720, "RealSense Align Example");
    ImGui_ImplGlfw_Init(app, false);
    rs2::colorizer c;
    texture depth_image, color_image;

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH);
    cfg.enable_stream(RS2_STREAM_COLOR);
    pipe.start(cfg);

    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    float alpha = .5f;
    direction dir = direction::to_depth;

    while (app) {
        rs2::frameset frameset = pipe.wait_for_frames();
        if (dir == direction::to_depth) {
            frameset = align_to_depth.process(frameset);
        } else {
            frameset = align_to_color.process(frameset);
        }

        auto depth = frameset.get_depth_frame();
        auto color = frameset.get_color_frame();
        auto colorized_depth = c.colorize(depth);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (dir == direction::to_depth) {
            depth_image.render(colorized_depth, {0, 0, app.width(), app.height()});
            color_image.render(color, {0, 0, app.width(), app.height()}, alpha);
        } else {
            color_image.render(color, { 0, 0, app.width(), app.height() });
            depth_image.render(colorized_depth, { 0, 0, app.width(), app.height() }, 1 - alpha);
        }

        glColor4f(1.f, 1.f, 1.f, 1.f);
        glDisable(GL_BLEND);

        // Render the UI:
        ImGui_ImplGlfw_NewFrame(1);
        render_slider({ 15.f, app.height() - 60, app.width() - 30, app.height() }, &alpha, &dir);
        ImGui::Render();
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

void render_slider(rect location, float* alpha, direction* dir)
{
    static const int flags = ImGuiWindowFlags_NoCollapse
        | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings
        | ImGuiWindowFlags_NoTitleBar
        | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos({ location.x, location.y });
    ImGui::SetNextWindowSize({ location.w, location.h });

    // Render transparency slider:
    ImGui::Begin("slider", nullptr, flags);
    ImGui::PushItemWidth(-1);
    ImGui::SliderFloat("##Slider", alpha, 0.f, 1.f);
    ImGui::PopItemWidth();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Texture Transparancy: %.3f", *alpha);

    // Render direction checkboxes:
    bool to_depth = (*dir == direction::to_depth);
    bool to_color = (*dir == direction::to_color);

    if (ImGui::Checkbox("Align To Depth", &to_depth))
    {
        *dir = to_depth ? direction::to_depth : direction::to_color;
    }
    ImGui::SameLine();
    ImGui::SetCursorPosX(location.w - 140);
    if (ImGui::Checkbox("Align To Color", &to_color))
    {
        *dir = to_color ? direction::to_color : direction::to_depth;
    }

    ImGui::End();
}