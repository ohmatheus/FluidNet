#pragma once

#include "SimulationBuffer.hpp"
#include <atomic>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <thread>

namespace FluidNet
{

class Simulation
{
public:
    Simulation();
    ~Simulation();

    void start();
    void stop();
    void restart();

    void setModel(const std::string& modelPath);
    void toggleGpuMode();

    const SimulationBuffer* getLatestState() const;

private:
    void workerLoop_();
    void runInferenceStep_(SimulationBuffer* frontBuf, SimulationBuffer* backBuf);
    void initializeOnnxSession_(const std::string& modelPath, bool useGpu);

    std::unique_ptr<Ort::Env> m_ortEnv;
    std::unique_ptr<Ort::Session> m_ortSession;
    std::unique_ptr<Ort::SessionOptions> m_sessionOpts;

    std::thread m_workerThread;
    std::atomic<bool> m_running{false};

    // Double buffering with pointers swap
    SimulationBuffer m_bufferA;
    SimulationBuffer m_bufferB;
    std::atomic<SimulationBuffer*> m_front;

    bool m_useGpu{true};
    std::string m_currentModelPath;
    float m_targetStepTime{0.0f};
};

}
