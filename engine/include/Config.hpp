#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace FluidNet
{

struct EngineConfig
{
    int windowWidth{1280};
    int windowHeight{720};
    bool gpuEnabled{true};
    std::vector<std::string> onnxProviders;
};

struct SimulationConfig
{
    float fps{30.0f};
    int gridResolution{128};
    int inputChannels{0};
};

struct ModelsConfig
{
    std::filesystem::path onnxFolder;
    int defaultIndex{0};
};

class Config
{
public:
    static Config& getInstance();

    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) = delete;
    Config& operator=(Config&&) = delete;

    void loadFromYaml(const std::string& path);

    int getWindowWidth() const
    {
        return m_engineConfig.windowWidth;
    }
    int getWindowHeight() const
    {
        return m_engineConfig.windowHeight;
    }
    bool isGpuEnabled() const
    {
        return m_engineConfig.gpuEnabled;
    }

    float getSimulationFPS() const
    {
        return m_simulationConfig.fps;
    }
    int getGridResolution() const
    {
        return m_simulationConfig.gridResolution;
    }
    int getInputChannels() const
    {
        return m_simulationConfig.inputChannels;
    }

    const std::filesystem::path& getModelsFolder() const
    {
        return m_resolvedModelsFolder;
    }
    int getDefaultModelIndex() const
    {
        return m_modelsConfig.defaultIndex;
    }

    const std::vector<std::string>& getOnnxProviders() const
    {
        return m_engineConfig.onnxProviders;
    }

private:
    Config() = default;
    ~Config() = default;

    EngineConfig m_engineConfig;
    SimulationConfig m_simulationConfig;
    ModelsConfig m_modelsConfig;
    std::filesystem::path m_resolvedModelsFolder;
};

}
