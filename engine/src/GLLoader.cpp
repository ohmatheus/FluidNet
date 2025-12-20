#include "GLLoader.hpp"

namespace FluidNet::GL
{

// Define all function pointers
PFNGLCREATESHADERPROC glCreateShader = nullptr;
PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
PFNGLDELETESHADERPROC glDeleteShader = nullptr;
PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
PFNGLATTACHSHADERPROC glAttachShader = nullptr;
PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;
PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
PFNGLBUFFERDATAPROC glBufferData = nullptr;
PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
PFNGLUNIFORM1IPROC glUniform1i = nullptr;
PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers = nullptr;
PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer = nullptr;
PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D = nullptr;
PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus = nullptr;
PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers = nullptr;

void loadGLFunctions()
{
    glCreateShader = (PFNGLCREATESHADERPROC)glfwGetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)glfwGetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)glfwGetProcAddress("glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)glfwGetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)glfwGetProcAddress("glGetShaderInfoLog");
    glDeleteShader = (PFNGLDELETESHADERPROC)glfwGetProcAddress("glDeleteShader");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)glfwGetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)glfwGetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)glfwGetProcAddress("glLinkProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)glfwGetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)glfwGetProcAddress("glGetProgramInfoLog");
    glDeleteProgram = (PFNGLDELETEPROGRAMPROC)glfwGetProcAddress("glDeleteProgram");
    glUseProgram = (PFNGLUSEPROGRAMPROC)glfwGetProcAddress("glUseProgram");
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)glfwGetProcAddress("glGenVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)glfwGetProcAddress("glBindVertexArray");
    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)glfwGetProcAddress("glDeleteVertexArrays");
    glGenBuffers = (PFNGLGENBUFFERSPROC)glfwGetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)glfwGetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)glfwGetProcAddress("glBufferData");
    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glfwGetProcAddress("glDeleteBuffers");
    glVertexAttribPointer =
        (PFNGLVERTEXATTRIBPOINTERPROC)glfwGetProcAddress("glVertexAttribPointer");
    glEnableVertexAttribArray =
        (PFNGLENABLEVERTEXATTRIBARRAYPROC)glfwGetProcAddress("glEnableVertexAttribArray");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)glfwGetProcAddress("glGetUniformLocation");
    glUniform1i = (PFNGLUNIFORM1IPROC)glfwGetProcAddress("glUniform1i");
    glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)glfwGetProcAddress("glGenFramebuffers");
    glBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC)glfwGetProcAddress("glBindFramebuffer");
    glFramebufferTexture2D =
        (PFNGLFRAMEBUFFERTEXTURE2DPROC)glfwGetProcAddress("glFramebufferTexture2D");
    glCheckFramebufferStatus =
        (PFNGLCHECKFRAMEBUFFERSTATUSPROC)glfwGetProcAddress("glCheckFramebufferStatus");
    glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)glfwGetProcAddress("glDeleteFramebuffers");
}

}
