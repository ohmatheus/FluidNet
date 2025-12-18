#include "Config.hpp"
#include "EngineConfig.hpp"
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>

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

        std::cout << "Window created. Press ESC to exit.\n";

        while (!glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);

            glfwSwapBuffers(window);

            glfwPollEvents();
        }

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