#include "Config.hpp"
#include "EngineConfig.hpp"
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

static void glfwErrorCallback(int error, const char* description)
{
    std::cerr << "GLFW Error " << error << ": " << description << "\n";
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main()
{
    try
    {
        std::filesystem::path configPath =
            std::filesystem::path(FluidNet::PROJECT_ROOT) / "config.yaml";
        auto config = FluidNet::EngineConfig::loadFromYaml(configPath.string());

        std::cout << "Configuration loaded successfully.\n";
        std::cout << "Window: " << config.window_width << "x" << config.window_height << "\n\n";

        glfwSetErrorCallback(glfwErrorCallback);

        if (!glfwInit())
        {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow* window = glfwCreateWindow(config.window_width, config.window_height, "FluidNet",
                                              nullptr, nullptr);

        if (!window)
        {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwSetKeyCallback(window, keyCallback);

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        glClearColor(0.1f, 0.15f, 0.2f, 1.0f);

        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        std::cout << "Window created. Press ESC to exit.\n";

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            // Start ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // ImGui demo window (shows all ImGui features)
            ImGui::ShowDemoWindow();

            // Custom debug window
            ImGui::Begin("FluidNet Engine");
            ImGui::Text("Configuration:");
            ImGui::Text("Window: %dx%d", config.window_width, config.window_height);
            ImGui::Text("FPS: %.1f", io.Framerate);
            ImGui::Separator();
            ImGui::Text("Press ESC to exit");
            ImGui::End();

            ImGui::Render();

            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();

        std::cout << "Engine terminated successfully.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        glfwTerminate();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}