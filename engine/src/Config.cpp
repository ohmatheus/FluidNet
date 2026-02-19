#include "Config.hpp"
#include "EngineConfig.hpp"
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace YAML
{

template <> struct convert<FluidNet::EngineConfig>
{
    static bool decode(const Node& node, FluidNet::EngineConfig& config)
    {
        if (!node["window_width"] || !node["window_height"] || !node["gpu_enabled"])
        {
            return false;
        }

        config.windowWidth = node["window_width"].as<int>();
        config.windowHeight = node["window_height"].as<int>();
        config.gpuEnabled = node["gpu_enabled"].as<bool>();

        config.onnxProviders.clear();
        if (node["onnx_providers"] && node["onnx_providers"].IsSequence())
        {
            for (const auto& provider : node["onnx_providers"])
            {
                config.onnxProviders.push_back(provider.as<std::string>());
            }
        }

        return true;
    }
};

template <> struct convert<FluidNet::SimulationConfig>
{
    static bool decode(const Node& node, FluidNet::SimulationConfig& config)
    {
        if (node["fps"])
        {
            config.fps = node["fps"].as<float>();
        }
        if (node["grid_resolution"])
        {
            config.gridResolution = node["grid_resolution"].as<int>();
        }
        if (node["input_channels"])
        {
            config.inputChannels = node["input_channels"].as<int>();
        }
        if (node["vorticity_default"])
        {
            config.vorticityDefault = node["vorticity_default"].as<float>();
        }
        if (node["vorticity_min"])
        {
            config.vorticityMin = node["vorticity_min"].as<float>();
        }
        if (node["vorticity_max"])
        {
            config.vorticityMax = node["vorticity_max"].as<float>();
        }
        if (node["vorticity_step"])
        {
            config.vorticityStep = node["vorticity_step"].as<float>();
        }
        return true;
    }
};

template <> struct convert<FluidNet::ModelsConfig>
{
    static bool decode(const Node& node, FluidNet::ModelsConfig& config)
    {
        if (node["onnx_folder"])
        {
            config.onnxFolder = node["onnx_folder"].as<std::string>();
        }
        if (node["default_index"])
        {
            config.defaultIndex = node["default_index"].as<int>();
        }
        return true;
    }
};

}

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

        if (!config["engine"])
        {
            throw std::runtime_error("Missing 'engine' section in config file");
        }

        m_engineConfig = config["engine"].as<EngineConfig>();

        if (config["simulation"])
        {
            m_simulationConfig = config["simulation"].as<SimulationConfig>();
        }

        if (config["models"])
        {
            m_modelsConfig = config["models"].as<ModelsConfig>();

            if (!m_modelsConfig.onnxFolder.empty())
            {
                m_resolvedModelsFolder = Paths::getProjectRoot() / m_modelsConfig.onnxFolder;
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
