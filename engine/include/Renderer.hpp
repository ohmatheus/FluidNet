#pragma once

#include "SimulationBuffer.hpp"

// GLFW includes OpenGL headers for us
#define GLFW_INCLUDE_NONE
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>
#include <string>

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

    void setDebugOverlay(bool enabled);
    void uploadSceneMasks(const std::vector<float>& emitterMask, const std::vector<float>& colliderMask, int gridRes);

    GLuint getFramebufferTexture() const
    {
        return m_framebufferTexture;
    }
    void resizeFramebuffer(int width, int height);

private:
    void uploadToGPU_(const SimulationBuffer& state);
    void compileShaders_();
    void createQuad_();
    void setupFramebuffer_();
    GLuint createTexture2D_(GLenum internalFormat, GLint minFilter = GL_LINEAR,
                            GLint magFilter = GL_LINEAR, GLint wrapS = GL_CLAMP_TO_EDGE,
                            GLint wrapT = GL_CLAMP_TO_EDGE);
    GLuint compileShader_(GLenum shaderType, const char* source, const std::string& shaderName);
    GLuint linkShaderProgram_(GLuint vertexShader, GLuint fragmentShader);

    int m_width;
    int m_height;

    GLuint m_vao{0};
    GLuint m_vbo{0};
    GLuint m_shaderProgram{0};
    GLuint m_debugShaderProgram{0};
    GLuint m_velocityTexture{0};
    GLuint m_densityTexture{0};
    GLuint m_emitterTexture{0};
    GLuint m_colliderTexture{0};

    GLuint m_framebuffer{0};
    GLuint m_framebufferTexture{0};
    int m_fbWidth{512};
    int m_fbHeight{512};

    bool m_initialized{false};
    bool m_showDebugOverlay{true};
};

}
