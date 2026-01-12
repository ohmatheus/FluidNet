#include "FluidScene.hpp"
#include "Config.hpp"
#include "ModelRegistry.hpp"
#include "SceneState.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <iostream>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

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

    m_sceneState = std::make_unique<SceneState>(config.getGridResolution());

    m_simulation = std::make_unique<Simulation>();
    m_simulation->setSceneSnapshot(m_sceneState->getSnapshotAtomic());

    if (m_modelRegistry && !m_modelRegistry->getModels().empty())
    {
        const auto& initialModel = m_modelRegistry->getCurrentModel();
        std::cout << "Loading initial model: " << initialModel.name << std::endl;
        m_simulation->setModel(initialModel.pathFP32);
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
#ifdef TRACY_ENABLE
    ZoneScopedN("Scene Update");
#endif

    if (m_simulation)
    {
        m_latestState = m_simulation->getLatestState();
    }

    if (m_sceneState)
    {
        float decayFactor = (100.0f - m_velocityDecayPercent) / 100.0f;
        m_sceneState->decayVelocityImpulses(decayFactor);
        m_sceneState->commitSnapshot();
    }
}

void FluidScene::render()
{
#ifdef TRACY_ENABLE
    ZoneScopedN("Scene Render");
#endif

    if (m_renderer && m_latestState)
    {
        if (m_sceneState)
        {
            const auto* snapshot = m_sceneState->getSnapshotAtomic()->load();
            if (snapshot)
            {
                m_renderer->uploadSceneMasks(snapshot->emitterMask, snapshot->colliderMask,
                                             snapshot->gridResolution);
            }
        }

        m_renderer->render(*m_latestState);
    }
}

void FluidScene::onRenderUI()
{
#ifdef TRACY_ENABLE
    ZoneScopedN("Fluid UI");
#endif

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

    const char* toolNames[] = {"Emitter", "Collider", "Velocity", "Erase"};
    ImGui::Text("Current Tool: %s", toolNames[static_cast<int>(m_currentTool)]);
    ImGui::SliderInt("Paint Brush", &m_paintBrushSize, 1, 15);
    ImGui::SliderInt("Velocity Brush", &m_velocityBrushSize, 1, 15);
    ImGui::SliderFloat("Velocity Strength", &m_velocityStrength, 0.1f, 1.0f, "%.1f");
    ImGui::SliderFloat("Velocity Decay %", &m_velocityDecayPercent, 1.0f, 10.0f, "%.1f%%");
    ImGui::Checkbox("Debug Overlay (O)", &m_showDebugOverlay);
    if (ImGui::IsItemEdited() && m_renderer)
    {
        m_renderer->setDebugOverlay(m_showDebugOverlay);
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

    ImGui::SameLine();

    if (ImGui::Button(m_isPaused ? "Resume (Space)" : "Pause (Space)"))
    {
        togglePause();
    }

    ImGui::Checkbox("Show Debug Info (D)", &m_showDebugInfo);

#ifdef TRACY_ENABLE
    if (ImGui::Checkbox("Enable Profiling (P)", &m_profilingEnabled))
    {
        std::cout << "Tracy profiling: " << (m_profilingEnabled ? "ENABLED" : "DISABLED")
                  << std::endl;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Toggle Tracy profiling on/off.\nPress 'P' or use this checkbox.");
    }
#endif

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
    ImGui::BulletText("E: Emitter tool");
    ImGui::BulletText("C: Collider tool");
    ImGui::BulletText("V: Velocity tool");
    ImGui::BulletText("X: Erase tool");
    ImGui::BulletText("O: Toggle debug overlay");
    ImGui::BulletText("D: Toggle debug info");
    ImGui::BulletText("M: Next model");
    ImGui::BulletText("Shift+M: Previous model");
    ImGui::BulletText("G: Toggle GPU/CPU");
    ImGui::BulletText("R: Restart simulation");
    ImGui::BulletText("Shift+R: Clear scene");
    ImGui::BulletText("Space: Pause/Resume");
    ImGui::BulletText("P: Toggle profiling");
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
        if (mods & GLFW_MOD_SHIFT)
        {
            if (m_sceneState)
            {
                m_sceneState->clear();
                m_sceneState->commitSnapshot();
                std::cout << "Scene cleared" << std::endl;
            }
        }
        else
        {
            restart();
        }
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

    case GLFW_KEY_O:
        if (m_renderer)
        {
            m_showDebugOverlay = !m_showDebugOverlay;
            m_renderer->setDebugOverlay(m_showDebugOverlay);
            std::cout << "Debug overlay: " << (m_showDebugOverlay ? "ON" : "OFF") << std::endl;
        }
        break;

    case GLFW_KEY_M:
        // This will be handled through Engine's model registry
        break;

    case GLFW_KEY_E:
        m_currentTool = Tool::Emitter;
        std::cout << "Tool: Emitter" << std::endl;
        break;

    case GLFW_KEY_C:
        m_currentTool = Tool::Collider;
        std::cout << "Tool: Collider" << std::endl;
        break;

    case GLFW_KEY_V:
        m_currentTool = Tool::Velocity;
        std::cout << "Tool: Velocity" << std::endl;
        break;

    case GLFW_KEY_X:
        m_currentTool = Tool::Erase;
        std::cout << "Tool: Erase" << std::endl;
        break;

    case GLFW_KEY_SPACE:
        togglePause();
        break;

    case GLFW_KEY_P:
        m_profilingEnabled = !m_profilingEnabled;
        std::cout << "Tracy profiling: " << (m_profilingEnabled ? "ENABLED" : "DISABLED")
                  << std::endl;
        break;
    }
}

void FluidScene::restart()
{
    if (m_simulation)
    {
        m_simulation->restart();
        if (!m_isPaused)
        {
            m_simulation->start();
        }
        std::cout << "Simulation restarted" << std::endl;
    }
}

void FluidScene::togglePause()
{
    m_isPaused = !m_isPaused;

    if (m_simulation)
    {
        if (m_isPaused)
        {
            m_simulation->stop();
            std::cout << "Simulation paused" << std::endl;
        }
        else
        {
            m_simulation->start();
            std::cout << "Simulation resumed" << std::endl;
        }
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

void FluidScene::handleMouseInput(float viewportX, float viewportY, float viewportWidth,
                                  float viewportHeight, bool leftButton, bool rightButton)
{
    if (!m_sceneState)
        return;

    float normalizedX = viewportX / viewportWidth;
    float normalizedY = viewportY / viewportHeight;

    int gridRes = m_sceneState->getGridResolution();
    int gridX = static_cast<int>(normalizedX * gridRes);
    int gridY = static_cast<int>(normalizedY * gridRes);

    if (gridX < 0 || gridX >= gridRes || gridY < 0 || gridY >= gridRes)
    {
        m_mousePressed = false;
        m_prevMouseGridX = -1;
        m_prevMouseGridY = -1;
        return;
    }

    if (leftButton)
    {
        switch (m_currentTool)
        {
        case Tool::Emitter:
            m_sceneState->paintEmitter(gridX, gridY, m_paintBrushSize);
            m_sceneState->commitSnapshot();
            break;
        case Tool::Collider:
            m_sceneState->paintCollider(gridX, gridY, m_paintBrushSize);
            m_sceneState->commitSnapshot();
            break;
        case Tool::Velocity:
            if (m_mousePressed && m_prevMouseGridX >= 0 && m_prevMouseGridY >= 0)
            {
                float deltaX = static_cast<float>(gridX - m_prevMouseGridX);
                float deltaY = static_cast<float>(gridY - m_prevMouseGridY);

                m_sceneState->paintVelocityImpulse(gridX, gridY, deltaX * m_velocityStrength,
                                                   deltaY * m_velocityStrength,
                                                   m_velocityBrushSize);
                m_sceneState->commitSnapshot();
            }
            m_mousePressed = true;
            m_prevMouseGridX = gridX;
            m_prevMouseGridY = gridY;
            break;
        case Tool::Erase:
            m_sceneState->erase(gridX, gridY, m_paintBrushSize);
            m_sceneState->commitSnapshot();
            break;
        }
    }
    else if (rightButton)
    {
        m_sceneState->erase(gridX, gridY, m_paintBrushSize);
        m_sceneState->commitSnapshot();
        m_mousePressed = false;
        m_prevMouseGridX = -1;
        m_prevMouseGridY = -1;
    }
    else
    {
        m_mousePressed = false;
        m_prevMouseGridX = -1;
        m_prevMouseGridY = -1;
    }
}

}
