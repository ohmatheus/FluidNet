#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace FluidNet
{

class Config
{
public:
    static Config& getInstance();

    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) = delete;
    Config& operator=(Config&&) = delete;

    void loadFromYaml(const std::string& path);

    // Engine settings
    int getWindowWidth() const
    {
        return m_windowWidth;
    }
    int getWindowHeight() const
    {
        return m_windowHeight;
    }
    bool isGpuEnabled() const
    {
        return m_gpuEnabled;
    }
    bool isVsyncEnabled() const
    {
        return m_vsyncEnabled;
    }

    int getGridResolution() const
    {
        return m_gridResolution;
    }
    float getSimulationFPS() const
    {
        return m_simulationFPS;
    }

    const std::filesystem::path& getModelsFolder() const
    {
        return m_modelsFolder;
    }
    int getDefaultModelIndex() const
    {
        return m_defaultModelIndex;
    }

    const std::vector<std::string>& getOnnxProviders() const
    {
        return m_onnxProviders;
    }

private:
    Config() = default;
    ~Config() = default;

    int m_windowWidth{1280};
    int m_windowHeight{720};
    bool m_gpuEnabled{true};
    bool m_vsyncEnabled{false};

    int m_gridResolution{128};
    float m_simulationFPS{30.0f};

    std::filesystem::path m_modelsFolder;
    int m_defaultModelIndex{0};

    std::vector<std::string> m_onnxProviders;
};

}
