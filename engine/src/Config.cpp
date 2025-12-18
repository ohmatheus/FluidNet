#include "Config.hpp"
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace FluidNet
{

EngineConfig EngineConfig::loadFromYaml(const std::string& path)
{
    try
    {
        YAML::Node config = YAML::LoadFile(path);

        EngineConfig engineConfig;

        // Load engine-specific settings
        if (config["engine"])
        {
            const auto& engine = config["engine"];
            engineConfig.window_width = engine["window_width"].as<int>();
            engineConfig.window_height = engine["window_height"].as<int>();
            engineConfig.gpu_enabled = engine["gpu_enabled"].as<bool>();

            // Load ONNX providers
            if (engine["onnx_providers"])
            {
                for (const auto& provider : engine["onnx_providers"])
                {
                    engineConfig.onnx_providers.push_back(provider.as<std::string>());
                }
            }
        }
        else
        {
            throw std::runtime_error("Missing 'engine' section in config file");
        }

        // Load general config settings
        if (config["general_config"])
        {
            const auto& general = config["general_config"];
            engineConfig.model_path = general["onnx_model_path"].as<std::string>();
            engineConfig.grid_resolution = general["grid_resolution"].as<int>();
        }
        else
        {
            throw std::runtime_error("Missing 'general_config' section in config file");
        }

        return engineConfig;
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
