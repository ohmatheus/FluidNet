#pragma once

#include <cstdint>
#include <vector>

namespace FluidNet
{

struct SimulationBuffer
{
    std::vector<float> density;
    std::vector<float> velocityX;
    std::vector<float> velocityY;
    std::vector<float> emitterMask;

    int gridResolution{128};
    uint64_t frameNumber{0};
    double timestamp{0.0};
    bool isDirty{false};

    void allocate(int resolution);
};

}
