#include "Simulation.hpp"
#include "Config.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <filesystem>
#include <iostream>

namespace
{
constexpr bool kEnableDebugDensityInjector = true;

// Continuously inject density in a circle at center-bottom of the grid
void injectDebugDensityCircle(FluidNet::SimulationBuffer& current,
                              FluidNet::SimulationBuffer& previous)
{
    if (!kEnableDebugDensityInjector)
    {
        return;
    }

    const int gridRes = current.gridResolution;
    if (gridRes <= 0)
    {
        return;
    }

    const float injectedDensity = 0.8f;
    const int radius = 20;
    const int centerX = gridRes / 2;
    const int centerY = gridRes / 2 + 30;

    const int radiusSq = radius * radius;

    const size_t planeSize = static_cast<size_t>(gridRes) * static_cast<size_t>(gridRes);
    if (current.density.size() != planeSize || previous.density.size() != planeSize)
    {
        return; // sizes not as expected, avoid UB in debug utility
    }

    for (int y = 0; y < gridRes; ++y)
    {
        const int dy = y - centerY;
        for (int x = 0; x < gridRes; ++x)
        {
            const int dx = x - centerX;
            const int distSq = dx * dx + dy * dy;
            if (distSq <= radiusSq)
            {
                const size_t idx =
                    static_cast<size_t>(y) * static_cast<size_t>(gridRes) + static_cast<size_t>(x);

                current.density[idx] += injectedDensity;
                previous.density[idx] += injectedDensity; // density_{t-1}
            }
        }
    }
}
} // namespace

namespace FluidNet
{

Simulation::Simulation()
{
    const auto& config = Config::getInstance();
    m_targetStepTime = 1.0f / config.getSimulationFPS();
    m_useGpu = config.isGpuEnabled();

    int resolution = config.getGridResolution();

    m_bufferA.allocate(resolution);
    m_bufferB.allocate(resolution);
    m_front.store(&m_bufferA, std::memory_order_release);

    // Initialize ONNX Runtime
    m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FluidNetEngine");
}

Simulation::~Simulation()
{
    stop();
}

void Simulation::initializeOnnxSession_(const std::string& modelPath, bool useGpu)
{
    try
    {
        m_sessionOpts = std::make_unique<Ort::SessionOptions>();
        m_sessionOpts->SetIntraOpNumThreads(1);
        m_sessionOpts->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Provider
        if (useGpu)
        {
            try
            {
                OrtCUDAProviderOptions cudaOpts{};
                m_sessionOpts->AppendExecutionProvider_CUDA(cudaOpts);
                std::cout << "Simulation: Using CUDA execution provider" << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cerr << "Warning: Failed to enable CUDA, falling back to CPU: " << e.what()
                          << std::endl;
            }
        }
        else
        {
            std::cout << "Simulation: Using CPU execution provider" << std::endl;
        }

        std::filesystem::path fullPath = std::filesystem::path(modelPath);
        if (!std::filesystem::exists(fullPath))
        {
            throw std::runtime_error("Model file not found: " + fullPath.string());
        }

        m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, fullPath.c_str(), *m_sessionOpts);
        m_currentModelPath = modelPath;

        std::cout << "Simulation: Loaded model: " << fullPath.filename() << std::endl;
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Failed to initialize ONNX session: " + std::string(e.what()));
    }
}

void Simulation::start()
{
    if (m_running)
    {
        return;
    }

    m_running = true;
    m_workerThread = std::thread(&Simulation::workerLoop_, this);
}

void Simulation::stop()
{
    if (!m_running)
    {
        return;
    }

    m_running = false;

    if (m_workerThread.joinable())
    {
        m_workerThread.join();
    }
}

void Simulation::restart()
{
    const auto& config = Config::getInstance();
    int resolution = config.getGridResolution();

    {
        m_bufferA.allocate(resolution);
        m_bufferB.allocate(resolution);

        m_front.store(&m_bufferA, std::memory_order_release);
    }
}

void Simulation::setModel(const std::string& modelPath)
{
    stop();

    try
    {
        initializeOnnxSession_(modelPath, m_useGpu);
        restart();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }

    start();
}

void Simulation::toggleGpuMode()
{
    m_useGpu = !m_useGpu;

    if (!m_currentModelPath.empty())
    {
        setModel(m_currentModelPath);
    }
}

const SimulationBuffer* Simulation::getLatestState() const
{
    // Atomic load: which buffer is currently "front"
    return m_front.load(std::memory_order_acquire);
}

float Simulation::getAvgComputeTimeMs() const
{
    return m_avgComputeTimeMs.load(std::memory_order_acquire);
}

void Simulation::workerLoop_()
{
    using Clock = std::chrono::high_resolution_clock;

    while (m_running)
    {
        auto stepStart = Clock::now();

        if (m_ortSession)
        {
            SimulationBuffer* frontBuf = m_front.load(std::memory_order_acquire);
            SimulationBuffer* backBuf = (frontBuf == &m_bufferA) ? &m_bufferB : &m_bufferA;

            // frontBuf:  density(t), vel(t)
            // backBuf:   density(t-1)
            // After this call, backBuf will contain t+1
            auto inferenceStart = Clock::now();
            runInferenceStep_(frontBuf, backBuf);
            auto inferenceEnd = Clock::now();

            m_front.store(backBuf, std::memory_order_release);

            // track compute time
            float inferenceMs =
                std::chrono::duration<float, std::milli>(inferenceEnd - inferenceStart).count();
            m_sumComputeTimeMs += inferenceMs;
            m_computeTimeSamples++;

            // every 1 second
            double currentTime = glfwGetTime();
            if (currentTime - m_lastAvgUpdate >= 1.0)
            {
                m_avgComputeTimeMs.store(m_sumComputeTimeMs / m_computeTimeSamples,
                                         std::memory_order_release);
                m_sumComputeTimeMs = 0.0f;
                m_computeTimeSamples = 0;
                m_lastAvgUpdate = currentTime;
            }
        }

        // fixed FPS
        auto elapsed = std::chrono::duration<float>(Clock::now() - stepStart).count();
        float sleepTime = m_targetStepTime - elapsed;

        if (sleepTime > 0.0f)
        {
            std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
        }
    }
}

void Simulation::runInferenceStep_(SimulationBuffer* frontBuf, SimulationBuffer* backBuf)
{
    // frontBuf->density: density_t
    // backBuf->density:  density_{t-1}
    try
    {
        const int gridRes = frontBuf->gridResolution;
        const size_t planeSize = static_cast<size_t>(gridRes) * static_cast<size_t>(gridRes);

        // Debug: continuously inject density in a circle
        injectDebugDensityCircle(*frontBuf, *backBuf);

        // Model input: (1, 4, H, W) = [density_t, velx_t, vely_t, density_{t-1}]
        const int64_t inputShape[] = {1, 4, gridRes, gridRes};
        const size_t inputSize = static_cast<size_t>(4) * planeSize;

        std::vector<float> inputData(inputSize);

        // Channel 0: density_t  (front)
        std::memcpy(&inputData[0 * planeSize], frontBuf->density.data(), planeSize * sizeof(float));

        // Channel 1: velx_t     (front)
        std::memcpy(&inputData[1 * planeSize], frontBuf->velocityX.data(),
                    planeSize * sizeof(float));

        // Channel 2: vely_t     (front)
        std::memcpy(&inputData[2 * planeSize], frontBuf->velocityY.data(),
                    planeSize * sizeof(float));

        // Channel 3: density_{t-1}  (back)
        std::memcpy(&inputData[3 * planeSize], backBuf->density.data(), planeSize * sizeof(float));

        // Create input tensor
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor =
            Ort::Value::CreateTensor<float>(memoryInfo, inputData.data(), inputSize, inputShape, 4);

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = m_ortSession->GetInputNameAllocated(0, allocator);
        auto outputName = m_ortSession->GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};

        auto outputTensors = m_ortSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor,
                                               1, outputNames, 1);

        // Extract output: [1, 3, H, W] = [density_{t+1}, velx_{t+1}, vely_{t+1}]
        if (!outputTensors.empty())
        {
            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            std::vector<int64_t> outputShape =
                outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

            // Basic sanity check (optional)
            if (outputShape.size() != 4 || outputShape[0] != 1 || outputShape[1] != 3 ||
                outputShape[2] != gridRes || outputShape[3] != gridRes)
            {
                std::cerr << "Unexpected ONNX output shape: [";
                for (size_t i = 0; i < outputShape.size(); ++i)
                {
                    std::cerr << outputShape[i] << (i + 1 < outputShape.size() ? "," : "");
                }
                std::cerr << "]\n";
                return;
            }

            // Ensure buffers have correct size
            if (backBuf->density.size() != planeSize)
            {
                backBuf->density.resize(planeSize);
            }
            if (backBuf->velocityX.size() != planeSize)
            {
                backBuf->velocityX.resize(planeSize);
            }
            if (backBuf->velocityY.size() != planeSize)
            {
                backBuf->velocityY.resize(planeSize);
            }

            for (size_t i = 0; i < planeSize; ++i)
            {
                backBuf->density[i] = outputData[0 * planeSize + i];   // density_{t+1}
                backBuf->velocityX[i] = outputData[1 * planeSize + i]; // velx_{t+1}
                backBuf->velocityY[i] = outputData[2 * planeSize + i]; // vely_{t+1}
            }

            // Metadata
            backBuf->frameNumber = frontBuf->frameNumber + 1;
            backBuf->timestamp = glfwGetTime(); // TODO: better timer
            backBuf->isDirty = true;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Inference error: " << e.what() << std::endl;
    }
}

}
