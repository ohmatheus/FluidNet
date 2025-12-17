#include "Config.hpp"
#include "EngineConfig.hpp"
#include <filesystem>
#include <iostream>

int main()
{
    try
    {
        std::filesystem::path configPath =
            std::filesystem::path(FluidNet::PROJECT_ROOT) / "config.yaml";
        auto config = FluidNet::EngineConfig::loadFromYaml(configPath.string());

        std::cout << "Configuration loaded successfully!\n\n";

        std::cout << "Window Settings:\n";
        std::cout << "  Width:  " << config.window_width << "\n";
        std::cout << "  Height: " << config.window_height << "\n\n";

        std::cout << "Model Settings:\n";
        std::cout << "  Path: " << config.model_path << "\n";
        std::cout << "  Grid Resolution: " << config.grid_resolution << "\n\n";

        std::cout << "ONNX Runtime Settings:\n";
        std::cout << "  GPU Enabled: " << (config.gpu_enabled ? "Yes" : "No") << "\n";
        std::cout << "  Providers:\n";
        for (const auto& provider : config.onnx_providers)
        {
            std::cout << "    - " << provider << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}