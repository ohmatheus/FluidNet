#pragma once

#include <GLFW/glfw3.h>
#include <memory>

namespace FluidNet
{

class Scene;
class ModelRegistry;

class Engine
{
public:
    Engine();
    ~Engine();

    void initialize();
    void run();
    void shutdown();

    void setScene(std::unique_ptr<Scene> scene);

    GLFWwindow* getWindow()
    {
        return m_window;
    }
    ModelRegistry& getModelRegistry()
    {
        return *m_modelRegistry;
    }

private:
    void processInput_();
    void renderFrame_();
    void setupDockspace_();
    void renderEngineDebugWindow_(float deltaTime);
    void renderViewportWindow_();
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void errorCallback(int error, const char* description);

    GLFWwindow* m_window{nullptr};
    std::unique_ptr<Scene> m_currentScene;
    std::unique_ptr<ModelRegistry> m_modelRegistry;

    // Timing
    double m_lastFrameTime{0.0};
};

}
