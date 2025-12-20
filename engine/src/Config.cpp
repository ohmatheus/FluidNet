#include "Config.hpp"
#include "EngineConfig.hpp"
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace FluidNet
{

Config& Config::getInstance()
{
    static Config instance;
    return instance;
}

void Config::loadFromYaml(const std::string& path)
{
    try
    {
        YAML::Node config = YAML::LoadFile(path);

        if (config["engine"])
        {
            const auto& engine = config["engine"];
            m_windowWidth = engine["window_width"].as<int>();
            m_windowHeight = engine["window_height"].as<int>();
            m_gpuEnabled = engine["gpu_enabled"].as<bool>();

            m_onnxProviders.clear();
            if (engine["onnx_providers"])
            {
                for (const auto& provider : engine["onnx_providers"])
                {
                    m_onnxProviders.push_back(provider.as<std::string>());
                }
            }
        }
        else
        {
            throw std::runtime_error("Missing 'engine' section in config file");
        }

        if (config["simulation"])
        {
            const auto& simulation = config["simulation"];

            if (simulation["fps"])
            {
                m_simulationFPS = simulation["fps"].as<float>();
            }

            if (simulation["grid_resolution"])
            {
                m_gridResolution = simulation["grid_resolution"].as<int>();
            }
        }

        if (config["models"])
        {
            const auto& models = config["models"];

            if (models["folder"])
            {
                m_modelsFolder = Paths::getProjectRoot() /
                                 std::filesystem::path(models["folder"].as<std::string>());
            }

            if (models["default_index"])
            {
                m_defaultModelIndex = models["default_index"].as<int>();
            }
        }
    }
    catch (const YAML::Exception& e)
    {
        throw std::runtime_error("Failed to parse YAML config: " + std::string(e.what()));
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Failed to load config: " + std::string(e.what()));
    }
}

}
