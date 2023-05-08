#include "gui.hpp"
#include "window.hpp"
#include <sthe/config/debug.hpp>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace sthe
{

GUI& getGUI()
{
    static GUI gui;
    return gui;
}

// Constructor
GUI::GUI()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext(); 
    
    ImGui::StyleColorsDark();
    ImGuiStyle& style{ ImGui::GetStyle() };
    ImVec4* const colors{ style.Colors };

    style.WindowBorderSize = 0;
    style.PopupBorderSize = 0;
    style.FrameBorderSize = 0;
    style.TabBorderSize = 0;
    style.WindowRounding = 0;
    style.TabRounding = 0;
    style.ScrollbarRounding = 0;
    style.PopupRounding = 0;
    style.GrabRounding = 0;
    style.FrameRounding = 0;
    style.ChildRounding = 0;
    style.WindowMenuButtonPosition = ImGuiDir_Left;

    colors[ImGuiCol_Text] = ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f };
    colors[ImGuiCol_TextDisabled] = ImVec4{ 0.5f, 0.5f, 0.5f, 1.0f };
    colors[ImGuiCol_WindowBg] = ImVec4{ 0.06f, 0.06f, 0.06f, 0.94f };
    colors[ImGuiCol_ChildBg] = ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f };
    colors[ImGuiCol_PopupBg] = ImVec4{ 0.08f, 0.08f, 0.08f, 0.94f };
    colors[ImGuiCol_Border] = ImVec4{ 0.43f, 0.43f, 0.5f, 0.5f };
    colors[ImGuiCol_BorderShadow] = ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f };
    colors[ImGuiCol_FrameBg] = ImVec4{ 0.44f, 0.44f, 0.44f, 0.6f };
    colors[ImGuiCol_FrameBgHovered] = ImVec4{ 0.57f, 0.57f, 0.57f, 0.7f };
    colors[ImGuiCol_FrameBgActive] = ImVec4{ 0.76f, 0.76f, 0.76f, 0.8f };
    colors[ImGuiCol_TitleBg] = ImVec4{ 0.04f, 0.04f, 0.04f, 1.0f };
    colors[ImGuiCol_TitleBgActive] = ImVec4{ 0.16f, 0.16f, 0.16f, 1.0f };
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4{ 0.0f, 0.0f, 0.0f, 0.6f };
    colors[ImGuiCol_MenuBarBg] = ImVec4{ 0.14f, 0.14f, 0.14f, 1.0f };
    colors[ImGuiCol_ScrollbarBg] = ImVec4{ 0.02f, 0.02f, 0.02f, 0.53f };
    colors[ImGuiCol_ScrollbarGrab] = ImVec4{ 0.31f, 0.31f, 0.31f, 1.0f };
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4{ 0.41f, 0.41f, 0.41f, 1.0f };
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4{ 0.51f, 0.51f, 0.51f, 1.0f };
    colors[ImGuiCol_CheckMark] = ImVec4{ 0.0f, 0.55f, 0.55f, 0.8f };
    colors[ImGuiCol_SliderGrab] = ImVec4{ 0.13f, 0.75f, 0.75f, 0.8f };
    colors[ImGuiCol_SliderGrabActive] = ImVec4{ 0.13f, 0.75f, 1.0f, 0.8f };
    colors[ImGuiCol_Button] = ImVec4{ 0.0f, 0.55f, 0.55f, 0.4f };
    colors[ImGuiCol_ButtonHovered] = ImVec4{ 0.13f, 0.75f, 0.75f, 0.6f };
    colors[ImGuiCol_ButtonActive] = ImVec4{ 0.13f, 0.75f, 1.0f, 0.8f };
    colors[ImGuiCol_Header] = ImVec4{ 0.0f, 0.55f, 0.55f, 0.4f };
    colors[ImGuiCol_HeaderHovered] = ImVec4{ 0.13f, 0.75f, 0.75f, 0.6f };
    colors[ImGuiCol_HeaderActive] = ImVec4{ 0.13f, 0.75f, 1.0f, 0.8f };
    colors[ImGuiCol_Separator] = ImVec4{ 0.0f, 0.55f, 0.55f, 0.4f };
    colors[ImGuiCol_SeparatorHovered] = ImVec4{ 0.13f, 0.75f, 0.75f, 0.6f };
    colors[ImGuiCol_SeparatorActive] = ImVec4{ 0.13f, 0.75f, 1.0f, 0.8f };
    colors[ImGuiCol_ResizeGrip] = ImVec4{ 0.0f, 0.55f, 0.55f, 0.4f };
    colors[ImGuiCol_ResizeGripHovered] = ImVec4{ 0.13f, 0.75f, 0.75f, 0.6f };
    colors[ImGuiCol_ResizeGripActive] = ImVec4{ 0.13f, 0.75f, 1.0f, 0.8f };
    colors[ImGuiCol_Tab] = ImVec4{ 0.0f, 0.55f, 0.55f, 0.8f };
    colors[ImGuiCol_TabHovered] = ImVec4{ 0.13f, 0.75f, 0.75f, 0.8f };
    colors[ImGuiCol_TabActive] = ImVec4{ 0.13f, 0.75f, 1.0f, 0.8f };
    colors[ImGuiCol_TabUnfocused] = ImVec4{ 0.18f, 0.18f, 0.18f, 1.0f };
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4{ 0.36f, 0.36f, 0.36f, 0.54f };
    colors[ImGuiCol_PlotLines] = ImVec4{ 0.61f, 0.61f, 0.61f, 1.0f };
    colors[ImGuiCol_PlotLinesHovered] = ImVec4{ 1.0f, 0.43f, 0.35f, 1.0f };
    colors[ImGuiCol_PlotHistogram] = ImVec4{ 0.9f, 0.7f, 0.0f, 1.0f };
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4{ 1.0f, 0.6f, 0.0f, 1.0f };
    colors[ImGuiCol_TableHeaderBg] = ImVec4{ 0.19f, 0.19f, 0.2f, 1.0f };
    colors[ImGuiCol_TableBorderStrong] = ImVec4{ 0.31f, 0.31f, 0.35f, 1.0f };
    colors[ImGuiCol_TableBorderLight] = ImVec4{ 0.23f, 0.23f, 0.25f, 1.0f };
    colors[ImGuiCol_TableRowBg] = ImVec4{ 0.0f, 0.0f, 0.0f, 0.0f };
    colors[ImGuiCol_TableRowBgAlt] = ImVec4{ 1.0f, 1.0f, 1.0f, 0.07f };
    colors[ImGuiCol_TextSelectedBg] = ImVec4{ 0.26f, 0.59f, 0.98f, 0.35f };
    colors[ImGuiCol_DragDropTarget] = ImVec4{ 1.0f, 1.0f, 0.0f, 0.9f };
    colors[ImGuiCol_NavHighlight] = ImVec4{ 0.26f, 0.59f, 0.98f, 1.0f };
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4{ 1.0f, 1.0f, 1.0f, 0.7f };
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4{ 0.8f, 0.8f, 0.8f, 0.2f };
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4{ 0.8f, 0.8f, 0.8f, 0.35f };

    [[maybe_unused]] bool status{ ImGui_ImplGlfw_InitForOpenGL(getWindow().getHandle(), true) };

    STHE_ASSERT(status, "Failed to initialize ImGui for GLFW");

    status = ImGui_ImplOpenGL3_Init("#version 460 core");

    STHE_ASSERT(status, "Failed to initialize ImGui for OpenGL");
}

// Destructor
GUI::~GUI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

// Functionality
void GUI::start()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GUI::render()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}
