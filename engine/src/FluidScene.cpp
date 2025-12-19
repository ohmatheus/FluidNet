#include "FluidScene.hpp"
#include "Config.hpp"
#include "ModelRegistry.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <iostream>

namespace FluidNet
{

FluidScene::FluidScene(ModelRegistry* registry) : m_modelRegistry(registry) {}

FluidScene::~FluidScene()
{
    onShutdown();
}

void FluidScene::onInit()
{
    const auto& config = Config::getInstance();

    m_simulation = std::make_unique<Simulation>();

    if (m_modelRegistry && !m_modelRegistry->getModels().empty())
    {
        const auto& initialModel = m_modelRegistry->getCurrentModel();
        std::cout << "Loading initial model: " << initialModel.name << std::endl;
        m_simulation->setModel(initialModel.path);
    }
    else
    {
        std::cout << "Warning: No models available in registry" << std::endl;
        m_simulation->start();
    }

    // Create renderer
    m_renderer = std::make_unique<Renderer>(config.getWindowWidth(), config.getWindowHeight());
    m_renderer->initialize();

    m_simulationFPS = config.getSimulationFPS();

    std::cout << "FluidScene initialized" << std::endl;
}

void FluidScene::onShutdown()
{
    if (m_simulation)
    {
        m_simulation->stop();
    }

    if (m_renderer)
    {
        m_renderer->shutdown();
    }

    m_simulation.reset();
    m_renderer.reset();

    std::cout << "FluidScene shutdown" << std::endl;
}

void FluidScene::onUpdate(float deltaTime)
{
    // Get latest simulation state
    if (m_simulation)
    {
        m_latestState = m_simulation->getLatestState();
    }
}

void FluidScene::render()
{
    // Render simulation state
    if (m_renderer && m_latestState)
    {
        m_renderer->render(*m_latestState);
    }
}

void FluidScene::onRenderUI()
{
    ImGui::Begin("Fluid Simulation");

    if (m_latestState)
    {
        ImGui::Text("Frame: %lu", m_latestState->frameNumber);
        ImGui::Text("Simulation FPS: %.1f (target)", m_simulationFPS);
        ImGui::Text("Grid Resolution: %d x %d", m_latestState->gridResolution,
                    m_latestState->gridResolution);
    }
    else
    {
        ImGui::Text("Waiting for simulation data...");
    }

    ImGui::Separator();

    if (ImGui::Button("Restart (R)"))
    {
        restart();
    }

    ImGui::SameLine();

    if (ImGui::Button("Toggle GPU (G)"))
    {
        if (m_simulation)
        {
            m_simulation->toggleGpuMode();
        }
    }

    ImGui::Checkbox("Show Debug Info (D)", &m_showDebugInfo);

    if (m_showDebugInfo && m_latestState)
    {
        ImGui::Separator();
        ImGui::Text("Timestamp: %.2f s", m_latestState->timestamp);

        // Calculate max values for debug
        float maxDensity = 0.0f;
        for (float d : m_latestState->density)
        {
            maxDensity = std::max(maxDensity, d);
        }

        ImGui::Text("Max Density: %.3f", maxDensity);
    }

    ImGui::Separator();
    ImGui::TextWrapped("Keyboard shortcuts:");
    ImGui::BulletText("M: Next model");
    ImGui::BulletText("Shift+M: Previous model");
    ImGui::BulletText("G: Toggle GPU/CPU");
    ImGui::BulletText("R: Restart simulation");
    ImGui::BulletText("D: Toggle debug info");
    ImGui::BulletText("ESC: Exit");

    ImGui::End();
}

void FluidScene::onKeyPress(int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS)
    {
        return;
    }

    switch (key)
    {
    case GLFW_KEY_R:
        restart();
        break;

    case GLFW_KEY_G:
        if (m_simulation)
        {
            m_simulation->toggleGpuMode();
        }
        break;

    case GLFW_KEY_D:
        m_showDebugInfo = !m_showDebugInfo;
        break;

    case GLFW_KEY_M:
        // This will be handled through Engine's model registry
        break;
    }
}

void FluidScene::restart()
{
    if (m_simulation)
    {
        m_simulation->restart();
        std::cout << "Simulation restarted" << std::endl;
    }
}

void FluidScene::onModelChanged(const std::string& modelPath)
{
    if (m_simulation)
    {
        m_simulation->setModel(modelPath);
        std::cout << "Model changed to: " << modelPath << std::endl;
    }
}

}
