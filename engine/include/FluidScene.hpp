#pragma once

#include "Renderer.hpp"
#include "Scene.hpp"
#include "Simulation.hpp"
#include "SimulationBuffer.hpp"
#include <memory>

namespace FluidNet
{

class ModelRegistry;

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

    const SimulationBuffer* m_latestState{nullptr};
    bool m_showDebugInfo{true};
    float m_simulationFPS{0.0f};

    ModelRegistry* m_modelRegistry{nullptr};
};

}
