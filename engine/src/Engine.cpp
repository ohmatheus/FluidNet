#include "Engine.hpp"
#include "Config.hpp"
#include "FluidScene.hpp"
#include "ModelRegistry.hpp"
#include "Scene.hpp"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <chrono>
#include <imgui.h>
#include <iostream>

void FluidNet::Engine::errorCallback(int error, const char* description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

void FluidNet::Engine::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    Engine* engine = static_cast<Engine*>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    if (engine->m_currentScene)
    {
        engine->m_currentScene->onKeyPress(key, scancode, action, mods);
    }
}

namespace FluidNet
{

Engine::Engine() : m_modelRegistry(std::make_unique<ModelRegistry>()) {}

Engine::~Engine()
{
    shutdown();
}

void Engine::initialize()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    const auto& config = Config::getInstance();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(config.getWindowWidth(), config.getWindowHeight(),
                                "FluidNet Engine", nullptr, nullptr);

    if (!m_window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetWindowUserPointer(m_window, this);

    glfwSetKeyCallback(m_window, keyCallback);

    glfwMakeContextCurrent(m_window);

    glfwSwapInterval(config.isVsyncEnabled() ? 1 : 0);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Initialize model registry
    try
    {
        m_modelRegistry->initialize(config.getModelsFolder());
        std::cout << "Model Registry: Found " << m_modelRegistry->getModels().size() << " models"
                  << std::endl;
        for (const auto& model : m_modelRegistry->getModels())
        {
            std::cout << "  - " << model.name << " (" << model.path << ")" << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Warning: Failed to initialize model registry: " << e.what() << std::endl;
    }

    glClearColor(0.1f, 0.15f, 0.2f, 1.0f);

    m_lastFrameTime = glfwGetTime();
}

void Engine::run()
{
    while (!glfwWindowShouldClose(m_window))
    {
        renderFrame_();
    }
}

void Engine::renderFrame_()
{
    double currentTime = glfwGetTime();
    float deltaTime = static_cast<float>(currentTime - m_lastFrameTime);
    m_lastFrameTime = currentTime;

    glfwPollEvents();

    if (m_currentScene)
    {
        m_currentScene->onUpdate(deltaTime);
    }

    // ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Engine");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("Frame Time: %.2f ms", deltaTime * 1000.0f);

    if (auto* fluidScene = dynamic_cast<FluidScene*>(m_currentScene.get()))
    {
        if (auto* sim = fluidScene->getSimulation())
        {
            ImGui::Text("Sim Compute: %.2f ms", sim->getAvgComputeTimeMs());
        }
    }

    // Model selector - maybe move this to scene
    if (!m_modelRegistry->getModels().empty())
    {
        const auto& models = m_modelRegistry->getModels();
        int currentIdx = m_modelRegistry->getCurrentIndex();

        if (ImGui::BeginCombo("Model", models[currentIdx].name.c_str()))
        {
            for (int i = 0; i < static_cast<int>(models.size()); ++i)
            {
                bool selected = (i == currentIdx);
                if (ImGui::Selectable(models[i].name.c_str(), selected))
                {
                    m_modelRegistry->selectModel(i);
                    // TODO: Notify scene of model change
                }
                if (selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    ImGui::End();

    if (m_currentScene)
    {
        m_currentScene->onRenderUI();
    }

    ImGui::Render();

    glClear(GL_COLOR_BUFFER_BIT);

    if (m_currentScene)
    {
        m_currentScene->render();
    }

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(m_window);
}

void Engine::setScene(std::unique_ptr<Scene> scene)
{
    if (m_currentScene)
    {
        m_currentScene->onShutdown();
    }

    m_currentScene = std::move(scene);
}

void Engine::shutdown()
{
    if (m_currentScene)
    {
        m_currentScene->onShutdown();
        m_currentScene.reset();
    }

    if (m_window)
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    if (m_window)
    {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    glfwTerminate();
}

}
