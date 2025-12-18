#include "Config.hpp"
#include "EngineConfig.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <onnxruntime_cxx_api.h>

static void glfwErrorCallback(int error, const char* description)
{
    std::cerr << "GLFW Error " << error << ": " << description << "\n";
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main()
{
    try
    {
        std::filesystem::path configPath =
            std::filesystem::path(FluidNet::PROJECT_ROOT) / "config.yaml";
        auto config = FluidNet::EngineConfig::loadFromYaml(configPath.string());

        std::cout << "Configuration loaded successfully.\n";
        std::cout << "Window: " << config.window_width << "x" << config.window_height << "\n";

        // Test ONNX
        std::cout << "\n--- ONNX Runtime Test ---\n";
        try
        {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FluidNetEngine");
            std::cout << "ONNX Runtime initialized successfully\n";

            // Providers
            std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
            std::cout << "Available providers (" << availableProviders.size() << "):\n";
            for (const auto& provider : availableProviders)
            {
                std::cout << "  - " << provider << "\n";
            }

            std::cout << "\nRequested providers from config:\n";
            for (const auto& provider : config.onnx_providers)
            {
                bool available = std::find(availableProviders.begin(), availableProviders.end(),
                                           provider) != availableProviders.end();
                std::cout << "  - " << provider << ": "
                          << (available ? "Available" : "Not available") << "\n";
            }

            // Session
            Ort::SessionOptions sessionOpts;
            sessionOpts.SetIntraOpNumThreads(1);
            sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            // CUDA provider ?
            bool cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(),
                                           "CUDAExecutionProvider") != availableProviders.end();
            bool cudaEnabled = false;
            if (config.gpu_enabled && cudaAvailable)
            {
                try
                {
                    OrtCUDAProviderOptions cudaOpts{};
                    sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
                    cudaEnabled = true;
                    std::cout << "\nCUDA provider enabled for inference\n";
                }
                catch (const Ort::Exception& e)
                {
                    std::cout << "\n!!! CUDA provider available but failed to initialize: "
                              << e.what() << "\n  Falling back to CPU\n";
                }
            }
            else if (config.gpu_enabled && !cudaAvailable)
            {
                std::cout << "\n!!! CUDA provider requested but not available, will use CPU\n";
            }

            if (!cudaEnabled)
            {
                std::cout << (config.gpu_enabled ? "" : "\n")
                          << "Using CPU provider for inference\n";
            }

            // Dummy inference test
            std::cout << "\n--- Running Dummy Inference Test ---\n";

            // Load model
            std::filesystem::path modelPath =
                std::filesystem::path(FluidNet::PROJECT_ROOT) / config.model_path;
            std::cout << "Loading model: " << modelPath << "\n";

            Ort::Session session(env, modelPath.c_str(), sessionOpts);
            std::cout << "Model loaded successfully\n";

            const int64_t batch_size = 1;
            const int64_t input_channels = 4;
            const int64_t height = config.grid_resolution;
            const int64_t width = config.grid_resolution;

            std::array<int64_t, 4> input_shape = {batch_size, input_channels, height, width};
            size_t input_tensor_size = batch_size * input_channels * height * width;
            std::vector<float> input_data(input_tensor_size, 0.5f); // Dummy data

            std::cout << "Input shape: [" << batch_size << ", " << input_channels << ", " << height
                      << ", " << width << "]\n";
            std::cout << "Input tensor size: " << input_tensor_size << " elements\n";

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor =
                Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                input_shape.data(), input_shape.size());

            // Run inference
            const char* input_names[] = {"input"};
            const char* output_names[] = {"output"};

            // Warm-up run (avoid 1st run initialisation time overhead in clock) - to remove later
            std::cout << "Warming up...\n";
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

            std::cout << "Running timed inference...\n";
            auto start = std::chrono::high_resolution_clock::now();

            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor,
                                              1, output_names, 1);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "Inference completed in " << duration.count() << " ms\n";

            // Extract output
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            std::cout << "Output shape: [";
            for (size_t i = 0; i < output_shape.size(); ++i)
            {
                std::cout << output_shape[i];
                if (i < output_shape.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";

            std::cout << "Device used: " << (cudaEnabled ? "GPU (CUDA)" : "CPU") << "\n";
            std::cout << "--- Dummy Inference Test OK ---\n\n";

            std::cout << "--- ONNX Runtime OK ---\n\n";
        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "ONNX Runtime error: " << e.what() << "\n";
            throw;
        }

        glfwSetErrorCallback(glfwErrorCallback);

        if (!glfwInit())
        {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow* window = glfwCreateWindow(config.window_width, config.window_height, "FluidNet",
                                              nullptr, nullptr);

        if (!window)
        {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwSetKeyCallback(window, keyCallback);

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        glClearColor(0.1f, 0.15f, 0.2f, 1.0f);

        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        std::cout << "Window created. Press ESC to exit.\n";

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            // Start ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // ImGui demo window (shows all ImGui features)
            ImGui::ShowDemoWindow();

            // Custom debug window
            ImGui::Begin("FluidNet Engine");
            ImGui::Text("Configuration:");
            ImGui::Text("Window: %dx%d", config.window_width, config.window_height);
            ImGui::Text("FPS: %.1f", io.Framerate);
            ImGui::Separator();
            ImGui::Text("Press ESC to exit");
            ImGui::End();

            ImGui::Render();

            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();

        std::cout << "Engine terminated successfully.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        glfwTerminate();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}