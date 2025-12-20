#pragma once

#include "SimulationBuffer.hpp"

// GLFW includes OpenGL headers for us
#define GLFW_INCLUDE_NONE
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>

namespace FluidNet
{

class Renderer
{
public:
    Renderer(int width, int height);
    ~Renderer();

    void initialize();
    void shutdown();

    void render(const SimulationBuffer& state);

    GLuint getFramebufferTexture() const
    {
        return m_framebufferTexture;
    }
    void resizeFramebuffer(int width, int height);

private:
    void uploadToGPU_(const SimulationBuffer& state);
    void compileShaders_();
    void createQuad_();

    int m_width;
    int m_height;

    GLuint m_vao{0};
    GLuint m_vbo{0};
    GLuint m_shaderProgram{0};
    GLuint m_velocityTexture{0};
    GLuint m_densityTexture{0};

    GLuint m_framebuffer{0};
    GLuint m_framebufferTexture{0};
    int m_fbWidth{512};
    int m_fbHeight{512};

    bool m_initialized{false};
};

}
