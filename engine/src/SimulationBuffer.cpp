#include "SimulationBuffer.hpp"

namespace FluidNet
{

void SimulationBuffer::allocate(int resolution)
{
    gridResolution = resolution;
    size_t size = resolution * resolution;

    density.resize(size, 0.0f);
    velocityX.resize(size, 0.0f);
    velocityY.resize(size, 0.0f);

    // todo: move those 2 as user input
    emitterMask.resize(size, 0.0f);
    colliderMask.resize(size, 0.0f);

    std::fill(density.begin(), density.end(), 0.0f);
    std::fill(velocityX.begin(), velocityX.end(), 0.0f);
    std::fill(velocityY.begin(), velocityY.end(), 0.0f);

    // todo: move those 2 as user input
    std::fill(emitterMask.begin(), emitterMask.end(), 0.0f);
    std::fill(colliderMask.begin(), colliderMask.end(), 0.0f);

    frameNumber = 0;
    timestamp = 0.0;
    isDirty = true;
}

}
