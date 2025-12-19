#include "Simulation.hpp"
#include "Config.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <filesystem>
#include <iostream>

namespace FluidNet
{

Simulation::Simulation()
{
    const auto& config = Config::getInstance();
    m_targetStepTime = 1.0f / config.getSimulationFPS();
    m_useGpu = config.isGpuEnabled();

    int resolution = config.getGridResolution();
    m_frontBuffer = std::make_unique<SimulationBuffer>();
    m_backBuffer = std::make_unique<SimulationBuffer>();
    m_frontBuffer->allocate(resolution);
    m_backBuffer->allocate(resolution);

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
        std::lock_guard<std::mutex> lock(m_bufferMutex);
        m_frontBuffer->allocate(resolution);
        m_backBuffer->allocate(resolution);
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
    // No lock needed - just returning a pointer
    // The pointer itself is atomic (won't change during read)
    return m_frontBuffer.get();
}

void Simulation::workerLoop_()
{
    using Clock = std::chrono::high_resolution_clock;

    while (m_running)
    {
        auto stepStart = Clock::now();

        if (m_ortSession)
        {
            runInferenceStep_();
        }

        // Swap buffers
        {
            std::lock_guard<std::mutex> lock(m_bufferMutex);
            std::swap(m_frontBuffer, m_backBuffer);
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

void Simulation::runInferenceStep_()
{
    try
    {
        // Prepare input tensor (dummy data for now)
        const int gridRes = m_backBuffer->gridResolution;
        const int64_t inputShape[] = {1, 4, gridRes, gridRes}; // [batch, channels, height, width]
        const size_t inputSize = 1 * 4 * gridRes * gridRes;

        std::vector<float> inputData(inputSize, 0.5f); // Dummy data

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

        // Extract output
        if (!outputTensors.empty())
        {
            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            std::vector<int64_t> outputShape =
                outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

            // Assuming output is [1, 3, gridRes, gridRes] for [density, velocityX, velocityY, ]
            const size_t planeSize = gridRes * gridRes;

            for (size_t i = 0; i < planeSize; ++i)
            {
                m_backBuffer->density[i] = outputData[i];
                m_backBuffer->velocityX[i] = outputData[planeSize + i];
                m_backBuffer->velocityY[i] = outputData[2 * planeSize + i];
            }

            m_backBuffer->frameNumber++;
            m_backBuffer->timestamp = glfwGetTime(); // TODO: Use proper timer
            m_backBuffer->isDirty = true;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Inference error: " << e.what() << std::endl;
    }
}

}
