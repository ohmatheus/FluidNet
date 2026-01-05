#include "Renderer.hpp"
#include "Config.hpp"
#include "EngineConfig.hpp"
#include "GLLoader.hpp"
#include <GL/gl.h>
#include <GL/glext.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace
{

std::string readShaderFile(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open shader file: " + filepath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
}

namespace FluidNet
{

Renderer::Renderer(int w, int h) : m_width(w), m_height(h) {}

Renderer::~Renderer()
{
    shutdown();
}

void Renderer::initialize()
{
    if (m_initialized)
    {
        return;
    }

    FluidNet::GL::loadGLFunctions();

    try
    {
        compileShaders_();
        createQuad_();

        m_velocityTexture = createTexture2D_(GL_RG32F);
        m_densityTexture = createTexture2D_(GL_R32F);
        m_emitterTexture = createTexture2D_(GL_R32F);
        m_colliderTexture = createTexture2D_(GL_R32F);

        setupFramebuffer_();

        m_initialized = true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Renderer initialization failed: " << e.what() << std::endl;
        shutdown();
        throw;
    }
}

void Renderer::shutdown()
{
    if (m_shaderProgram)
    {
        FluidNet::GL::glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }

    if (m_debugShaderProgram)
    {
        FluidNet::GL::glDeleteProgram(m_debugShaderProgram);
        m_debugShaderProgram = 0;
    }

    if (m_vao)
    {
        FluidNet::GL::glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_vbo)
    {
        FluidNet::GL::glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }

    if (m_velocityTexture)
    {
        glDeleteTextures(1, &m_velocityTexture);
        m_velocityTexture = 0;
    }

    if (m_densityTexture)
    {
        glDeleteTextures(1, &m_densityTexture);
        m_densityTexture = 0;
    }

    if (m_emitterTexture)
    {
        glDeleteTextures(1, &m_emitterTexture);
        m_emitterTexture = 0;
    }

    if (m_colliderTexture)
    {
        glDeleteTextures(1, &m_colliderTexture);
        m_colliderTexture = 0;
    }

    if (m_framebuffer)
    {
        FluidNet::GL::glDeleteFramebuffers(1, &m_framebuffer);
        m_framebuffer = 0;
    }

    if (m_framebufferTexture)
    {
        glDeleteTextures(1, &m_framebufferTexture);
        m_framebufferTexture = 0;
    }

    m_initialized = false;
}

GLuint Renderer::createTexture2D_(GLenum internalFormat, GLint minFilter, GLint magFilter,
                                  GLint wrapS, GLint wrapT)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
    return texture;
}

void Renderer::setupFramebuffer_()
{
    FluidNet::GL::glGenFramebuffers(1, &m_framebuffer);
    FluidNet::GL::glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);

    m_framebufferTexture = createTexture2D_(GL_RGB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_fbWidth, m_fbHeight, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 nullptr);
    FluidNet::GL::glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                         m_framebufferTexture, 0);

    if (FluidNet::GL::glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        throw std::runtime_error("Framebuffer not complete!");
    }

    FluidNet::GL::glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint Renderer::compileShader_(GLenum shaderType, const char* source,
                                const std::string& shaderName)
{
    GLuint shader = FluidNet::GL::glCreateShader(shaderType);
    FluidNet::GL::glShaderSource(shader, 1, &source, nullptr);
    FluidNet::GL::glCompileShader(shader);

    GLint success;
    FluidNet::GL::glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        FluidNet::GL::glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        FluidNet::GL::glDeleteShader(shader);
        throw std::runtime_error(shaderName +
                                 " shader compilation failed: " + std::string(infoLog));
    }

    return shader;
}

GLuint Renderer::linkShaderProgram_(GLuint vertexShader, GLuint fragmentShader)
{
    GLuint program = FluidNet::GL::glCreateProgram();
    FluidNet::GL::glAttachShader(program, vertexShader);
    FluidNet::GL::glAttachShader(program, fragmentShader);
    FluidNet::GL::glLinkProgram(program);

    GLint success;
    FluidNet::GL::glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        FluidNet::GL::glGetProgramInfoLog(program, 512, nullptr, infoLog);
        FluidNet::GL::glDeleteShader(vertexShader);
        FluidNet::GL::glDeleteShader(fragmentShader);
        throw std::runtime_error("Shader program linking failed: " + std::string(infoLog));
    }

    return program;
}

void Renderer::compileShaders_()
{
    std::filesystem::path shaderDir = Paths::getShaderDir();
    std::filesystem::path vertPath = shaderDir / "fluid.vert";
    std::filesystem::path fragPath = shaderDir / "fluid.frag";
    std::filesystem::path debugFragPath = shaderDir / "fluid_debug.frag";

    std::string vertexShaderStr = readShaderFile(vertPath.string());
    std::string fragmentShaderStr = readShaderFile(fragPath.string());
    std::string debugFragmentShaderStr = readShaderFile(debugFragPath.string());

    GLuint vertexShader = compileShader_(GL_VERTEX_SHADER, vertexShaderStr.c_str(), "Vertex");
    GLuint fragmentShader =
        compileShader_(GL_FRAGMENT_SHADER, fragmentShaderStr.c_str(), "Fragment");
    GLuint debugFragmentShader =
        compileShader_(GL_FRAGMENT_SHADER, debugFragmentShaderStr.c_str(), "Debug Fragment");

    m_shaderProgram = linkShaderProgram_(vertexShader, fragmentShader);

    m_debugShaderProgram = linkShaderProgram_(vertexShader, debugFragmentShader);

    FluidNet::GL::glDeleteShader(vertexShader);
    FluidNet::GL::glDeleteShader(fragmentShader);
}

void Renderer::createQuad_()
{
    // quad, position & flipped texture coordinates (v -> 1 - v)
    float vertices[] = {
        -1.0f, 1.0f,  0.0f, 0.0f, // top-left  -> (0,0)
        -1.0f, -1.0f, 0.0f, 1.0f, // bottom-left -> (0,1)
        1.0f,  -1.0f, 1.0f, 1.0f, // bottom-right -> (1,1)
        -1.0f, 1.0f,  0.0f, 0.0f, // top-left
        1.0f,  -1.0f, 1.0f, 1.0f, // bottom-right
        1.0f,  1.0f,  1.0f, 0.0f  // top-right -> (1,0)
    };

    FluidNet::GL::glGenVertexArrays(1, &m_vao);
    FluidNet::GL::glGenBuffers(1, &m_vbo);
    FluidNet::GL::glBindVertexArray(m_vao);
    FluidNet::GL::glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    FluidNet::GL::glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Position attribute
    FluidNet::GL::glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    FluidNet::GL::glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    FluidNet::GL::glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                                        (void*)(2 * sizeof(float)));
    FluidNet::GL::glEnableVertexAttribArray(1);

    FluidNet::GL::glBindVertexArray(0);
}

void Renderer::uploadToGPU_(const SimulationBuffer& state)
{
    int res = state.gridResolution;

    // Upload velocity as RG texture
    std::vector<float> velocityRG(res * res * 2);
    for (int i = 0; i < res * res; ++i)
    {
        velocityRG[i * 2 + 0] = state.velocityX[i];
        velocityRG[i * 2 + 1] = state.velocityY[i];
    }

    glBindTexture(GL_TEXTURE_2D, m_velocityTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, res, res, 0, GL_RG, GL_FLOAT, velocityRG.data());

    std::vector<float> densityVis = state.density;

    // Upload density as R texture
    glBindTexture(GL_TEXTURE_2D, m_densityTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, res, res, 0, GL_RED, GL_FLOAT, densityVis.data());
}

void Renderer::uploadSceneMasks(const std::vector<float>& emitterMask,
                                 const std::vector<float>& colliderMask, int gridRes)
{
    if (!m_initialized)
    {
        return;
    }

    glBindTexture(GL_TEXTURE_2D, m_emitterTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, gridRes, gridRes, 0, GL_RED, GL_FLOAT,
                 emitterMask.data());

    glBindTexture(GL_TEXTURE_2D, m_colliderTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, gridRes, gridRes, 0, GL_RED, GL_FLOAT,
                 colliderMask.data());
}

void Renderer::setDebugOverlay(bool enabled)
{
    m_showDebugOverlay = enabled;
}

void Renderer::render(const SimulationBuffer& state)
{
    if (!m_initialized)
    {
        return;
    }

    if (state.isDirty)
    {
        uploadToGPU_(state);
    }

    // Bind framebuffer for offscreen rendering
    FluidNet::GL::glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
    glViewport(0, 0, m_fbWidth, m_fbHeight);
    glClearColor(0, 0.3, 0.5, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint activeProgram = m_showDebugOverlay ? m_debugShaderProgram : m_shaderProgram;
    FluidNet::GL::glUseProgram(activeProgram);

    // Bind density texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_densityTexture);
    FluidNet::GL::glUniform1i(FluidNet::GL::glGetUniformLocation(activeProgram, "densityTexture"),
                              0);

    if (m_showDebugOverlay)
    {
        // Bind emitter texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, m_emitterTexture);
        FluidNet::GL::glUniform1i(
            FluidNet::GL::glGetUniformLocation(activeProgram, "emitterTexture"), 1);

        // Bind collider texture
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, m_colliderTexture);
        FluidNet::GL::glUniform1i(
            FluidNet::GL::glGetUniformLocation(activeProgram, "colliderTexture"), 2);
    }

    FluidNet::GL::glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    FluidNet::GL::glBindVertexArray(0);

    // Unbind framebuffer back to screen
    FluidNet::GL::glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::resizeFramebuffer(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        return;
    }

    m_fbWidth = width;
    m_fbHeight = height;

    glBindTexture(GL_TEXTURE_2D, m_framebufferTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

}
