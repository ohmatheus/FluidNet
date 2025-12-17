#pragma once

#include <string>
#include <vector>

namespace FluidNet
{

struct EngineConfig
{
    int window_width;
    int window_height;

    std::string model_path;
    int grid_resolution;

    std::vector<std::string> onnx_providers;
    bool gpu_enabled;

    static EngineConfig loadFromYaml(const std::string& path);
};

}
