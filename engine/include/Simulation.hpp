#pragma once

#include "SceneState.hpp"
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

    void setModel(const std::string& modelPath, bool forceReload = false);
    void toggleGpuMode();
    bool isUsingCpu() const
    {
        return !m_useGpu;
    }

    const SimulationBuffer* getLatestState() const;
    float getAvgComputeTimeMs() const;

    void setSceneSnapshot(const std::atomic<SceneMaskSnapshot*>* snapshot);

private:
    void workerLoop_();
    float runInferenceStep_(SimulationBuffer* frontBuf, SimulationBuffer* backBuf,
                            const SceneMaskSnapshot* sceneSnapshot);
    void initializeOnnxSession_(const std::string& modelPath, bool useGpu);

    std::unique_ptr<Ort::Env> m_ortEnv;
    std::unique_ptr<Ort::Session> m_ortSession;
    std::unique_ptr<Ort::SessionOptions> m_sessionOpts;

    std::thread m_workerThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_restartRequested{false};

    // Double buffering with pointers swap
    SimulationBuffer m_bufferA;
    SimulationBuffer m_bufferB;
    std::atomic<SimulationBuffer*> m_front;

    bool m_useGpu{true};
    std::string m_currentModelPath;
    float m_targetStepTime{0.0f};

    const std::atomic<SceneMaskSnapshot*>* m_sceneSnapshotPtr{nullptr};

    std::atomic<float> m_avgComputeTimeMs{0.0f};
    float m_sumComputeTimeMs{0.0f};
    int m_computeTimeSamples{0};
    double m_lastAvgUpdate{0.0};
};

}
