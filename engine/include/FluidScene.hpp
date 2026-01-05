#pragma once

#include "Renderer.hpp"
#include "Scene.hpp"
#include "SceneState.hpp"
#include "Simulation.hpp"
#include "SimulationBuffer.hpp"
#include <memory>

namespace FluidNet
{

class ModelRegistry;

enum class Tool
{
    Emitter,
    Collider,
    Velocity,
    Erase
};

class FluidScene final : public Scene
{
public:
    FluidScene(ModelRegistry* modelRegistry = nullptr);
    ~FluidScene() override;

    void onInit() override;
    void onShutdown() override;
    void onUpdate(float deltaTime) override;
    void render() override;
    void onRenderUI() override;
    void onKeyPress(int key, int scancode, int action, int mods) override;
    void restart() override;

    void onModelChanged(const std::string& modelPath);
    void handleMouseInput(float viewportX, float viewportY, float viewportWidth,
                          float viewportHeight, bool leftButton, bool rightButton);

    Simulation* getSimulation() const
    {
        return m_simulation.get();
    }

    Renderer* getRenderer() const
    {
        return m_renderer.get();
    }

private:
    std::unique_ptr<Simulation> m_simulation;
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<SceneState> m_sceneState;

    const SimulationBuffer* m_latestState{nullptr};
    bool m_showDebugInfo{true};
    bool m_showDebugOverlay{true};
    float m_simulationFPS{0.0f};

    Tool m_currentTool{Tool::Emitter};
    int m_brushSize{2};

    int m_prevMouseGridX{-1};
    int m_prevMouseGridY{-1};
    bool m_mousePressed{false};

    ModelRegistry* m_modelRegistry{nullptr};
};

}
