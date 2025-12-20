#include "Config.hpp"
#include "Engine.hpp"
#include "EngineConfig.hpp"
#include "FluidScene.hpp"
#include <filesystem>
#include <iostream>
#include <memory>

int main()
{
    try
    {
        FluidNet::Config::getInstance().loadFromYaml(FluidNet::Paths::getConfigFile().string());
        std::cout << "Configuration loaded successfully." << std::endl;

        FluidNet::Engine engine;
        engine.initialize();

        auto scene = std::make_unique<FluidNet::FluidScene>(&engine.getModelRegistry());
        scene->onInit();

        engine.setScene(std::move(scene));
        engine.run();

        engine.shutdown();

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
