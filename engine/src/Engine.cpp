#include "Engine.hpp"
#include "Config.hpp"
#include "FluidScene.hpp"
#include "GLLoader.hpp"
#include "ModelRegistry.hpp"
#include "Profiling.hpp"
#include "Scene.hpp"
#include <GL/gl.h>
#include <GL/glext.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <chrono>
#include <imgui.h>
#include <iostream>

namespace FluidNet
{

Engine::Engine()
    : m_modelRegistry(std::make_unique<ModelRegistry>()), m_gpuPrecision(ModelPrecision::FP32)
{
}

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

    FluidNet::GL::loadGLFunctions();

    // no need for vsync
    glfwSwapInterval(0);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

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
            std::cout << "  - " << model.name << " (" << model.pathFP32 << ")" << std::endl;
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
    PROFILE_SET_THREAD_NAME("Main Thread");

    while (!glfwWindowShouldClose(m_window))
    {
        renderFrame_();

        FluidScene* fluidScene = dynamic_cast<FluidScene*>(m_currentScene.get());
        if (!fluidScene || fluidScene->isProfilingEnabled())
        {
            PROFILE_FRAME_MARK();
        }
    }
}

void Engine::setupDockspace_()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::Begin("DockSpace", nullptr, window_flags);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    ImGuiID dockspace_id = ImGui::GetID("MainDockspace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f));
    ImGui::End();
}

static const char* getPrecisionName(ModelPrecision precision)
{
    switch (precision)
    {
    case ModelPrecision::FP32:
        return "FP32";
    case ModelPrecision::FP16:
        return "FP16";
    case ModelPrecision::INT8:
        return "INT8";
    default:
        return "Unknown";
    }
}

void Engine::renderEngineDebugWindow_(float deltaTime)
{
    PROFILE_SCOPE_NAMED("Engine Debug UI");

    ImGui::Begin("Engine");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("Frame Time: %.2f ms", deltaTime * 1000.0f);

    if (auto* fluidScene = dynamic_cast<FluidScene*>(m_currentScene.get()))
    {
        if (auto* sim = fluidScene->getSimulation())
        {
            ImGui::Text("Sim Compute: %.2f ms", sim->getAvgComputeTimeMs());

            // Provider display
            bool usingCpu = sim->isUsingCpu();
            ImGui::Text("Provider: %s", usingCpu ? "CPU" : "CUDA");

            // Precision combo (only for GPU, CPU auto-uses INT8)
            if (!usingCpu)
            {
                ImGui::Separator();
                ImGui::Text("GPU Precision:");

                if (ImGui::BeginCombo("##Precision", getPrecisionName(m_gpuPrecision)))
                {
                    // Show available GPU precisions (FP32, FP16)
                    const auto& models = m_modelRegistry->getModels();
                    int currentIdx = m_modelRegistry->getCurrentIndex();
                    auto availablePrecisions = m_modelRegistry->getAvailablePrecisions(currentIdx);

                    for (ModelPrecision prec : availablePrecisions)
                    {
                        // Skip INT8 for GPU (it's CPU-only)
                        if (prec == ModelPrecision::INT8)
                            continue;

                        bool selected = (prec == m_gpuPrecision);
                        if (ImGui::Selectable(getPrecisionName(prec), selected))
                        {
                            if (prec != m_gpuPrecision)
                            {
                                m_gpuPrecision = prec;

                                // Reload model with new precision
                                std::string modelPath =
                                    m_modelRegistry->getCurrentModelPath(m_gpuPrecision);
                                fluidScene->onModelChanged(modelPath);
                            }
                        }
                        if (selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                    }

                    ImGui::EndCombo();
                }
            }
        }
    }

    ImGui::Separator();

    if (!m_modelRegistry->getModels().empty())
    {
        const auto& models = m_modelRegistry->getModels();
        int currentIdx = m_modelRegistry->getCurrentIndex();

        if (ImGui::BeginCombo("Model", models[currentIdx].displayName.c_str()))
        {
            std::string currentCategory = "";

            for (int i = 0; i < static_cast<int>(models.size()); ++i)
            {
                if (models[i].relativeDir != currentCategory)
                {
                    if (i > 0)
                    {
                        ImGui::Separator();
                    }

                    if (!models[i].relativeDir.empty())
                    {
                        ImGui::TextDisabled("%s", models[i].relativeDir.c_str());
                    }

                    currentCategory = models[i].relativeDir;
                }

                bool selected = (i == currentIdx);

                // Display with indentation for nested folders
                int depth =
                    std::count(models[i].relativeDir.begin(), models[i].relativeDir.end(), '/');
                std::string indent(depth * 2, ' ');
                std::string displayText = indent + models[i].name;

                // Show precision indicators - May remove that
                std::string precisionInfo;
                if (models[i].hasFP16Variant && models[i].hasINT8Variant)
                {
                    precisionInfo = " [FP16+INT8]";
                }
                else if (models[i].hasFP16Variant)
                {
                    precisionInfo = " [FP16]";
                }
                else if (models[i].hasINT8Variant)
                {
                    precisionInfo = " [INT8]";
                }
                displayText += precisionInfo;

                if (ImGui::Selectable(displayText.c_str(), selected))
                {
                    if (i != currentIdx)
                    {
                        m_modelRegistry->selectModel(i);

                        if (auto* fluidScene = dynamic_cast<FluidScene*>(m_currentScene.get()))
                        {
                            if (auto* sim = fluidScene->getSimulation())
                            {
                                // Auto-select INT8 for CPU
                                ModelPrecision targetPrecision =
                                    sim->isUsingCpu() ? ModelPrecision::INT8 : m_gpuPrecision;

                                std::string modelPath =
                                    m_modelRegistry->getCurrentModelPath(targetPrecision);
                                fluidScene->onModelChanged(modelPath);
                            }
                        }
                    }
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
}

void Engine::renderViewportWindow_()
{
    PROFILE_SCOPE_NAMED("Viewport UI");

    ImGui::SetNextWindowSize(ImVec2(800.0f, 800.0f), ImGuiCond_Always);
    ImGui::Begin("Viewport");
    ImVec2 viewportSize = ImVec2(800.0f, 800.0f);

    if (m_currentScene)
    {
        if (auto* fluidScene = dynamic_cast<FluidScene*>(m_currentScene.get()))
        {
            if (auto* renderer = fluidScene->getRenderer())
            {
                renderer->resizeFramebuffer((int)viewportSize.x, (int)viewportSize.y);

                m_currentScene->render();

                GLuint texID = renderer->getFramebufferTexture();
                ImGui::Image((ImTextureID)(intptr_t)texID, viewportSize, ImVec2(0, 1),
                             ImVec2(1, 0));

                // Handle mouse input for simulation
                if (ImGui::IsItemHovered())
                {
                    ImVec2 mousePos = ImGui::GetMousePos();
                    ImVec2 imageMin = ImGui::GetItemRectMin();
                    ImVec2 imageMax = ImGui::GetItemRectMax();
                    ImVec2 relativePos = ImVec2(mousePos.x - imageMin.x, mousePos.y - imageMin.y);
                    ImVec2 imageSize = ImVec2(imageMax.x - imageMin.x, imageMax.y - imageMin.y);

                    bool leftButton = ImGui::IsMouseDown(ImGuiMouseButton_Left);
                    bool rightButton = ImGui::IsMouseDown(ImGuiMouseButton_Right);

                    fluidScene->handleMouseInput(relativePos.x, relativePos.y, imageSize.x,
                                                 imageSize.y, leftButton, rightButton);
                }
            }
        }
    }
    ImGui::End();
}

void Engine::renderFrame_()
{
    PROFILE_SCOPE();

    double currentTime = glfwGetTime();
    float deltaTime = static_cast<float>(currentTime - m_lastFrameTime);
    m_lastFrameTime = currentTime;

    glfwPollEvents();

    if (m_currentScene)
    {
        m_currentScene->onUpdate(deltaTime);
    }

    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    setupDockspace_();

    renderEngineDebugWindow_(deltaTime);

    if (m_currentScene)
    {
        m_currentScene->onRenderUI();
    }

    renderViewportWindow_();

    // Render ImGui
    ImGui::Render();

    // Clear screen and render ImGui to screen
    FluidNet::GL::glBindFramebuffer(GL_FRAMEBUFFER, 0);
    int display_w, display_h;
    glfwGetFramebufferSize(m_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClear(GL_COLOR_BUFFER_BIT);

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
