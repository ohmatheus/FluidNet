cmake_minimum_required(VERSION 3.20)

project(FluidNetEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clang-tidy

# Fetch yaml-cpp dependency
include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0  # Use specific version tag for reproducibility
)

# Configure yaml-cpp build
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib")
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "Build yaml-cpp as static library")

FetchContent_MakeAvailable(yaml-cpp)

# Fetch GLFW3 dependency
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)

# Configure GLFW build
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Disable GLFW tests")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Disable GLFW examples")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Disable GLFW documentation")
set(GLFW_INSTALL OFF CACHE BOOL "Disable GLFW installation")
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "Disable Wayland support, use X11 only")

FetchContent_MakeAvailable(glfw)

# Fetch ImGui dependency
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.8-docking  # Let's try out docks
)

FetchContent_MakeAvailable(imgui)

# Build ImGui as a static library
set(IMGUI_DIR ${imgui_SOURCE_DIR})

add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# ImGui backends need GLFW and OpenGL headers
target_link_libraries(imgui PUBLIC
    glfw
    OpenGL::GL
)

# Find OpenGL (system library)
find_package(OpenGL REQUIRED)

# Find Threads library (required for std::thread)
find_package(Threads REQUIRED)

# List source files explicitly for better build control
set(SOURCES
    src/main.cpp
    src/Config.cpp
    src/Engine.cpp
    src/FluidScene.cpp
    src/Simulation.cpp
    src/SimulationBuffer.cpp
    src/Renderer.cpp
    src/ModelRegistry.cpp
)

add_executable(fluid_engine ${SOURCES})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/EngineConfig.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/EngineConfig.hpp
)

target_include_directories(fluid_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

# ONNX Runtime (manually downloaded to external/onnxruntime)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime")

# Check if ONNX Runtime exists
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime not found in ${ONNXRUNTIME_ROOT}\n"
        "Please download and extract ONNX Runtime 1.23.2:\n"
        "  Linux:   https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz\n"
        "  Windows: https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip\n"
        "  macOS:   https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-arm64-1.23.2.tgz (Apple Silicon)\n"
        "           https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-x86_64-1.23.2.tgz (Intel)\n"
        "Extract to: engine/external/onnxruntime/\n"
        "See engine/README.md for detailed instructions.\n"
    )
endif()

add_library(onnxruntime SHARED IMPORTED)

if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
elseif(APPLE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
else() # Linux
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
endif()

# Set RPATH so executable finds the library at runtime
set_target_properties(fluid_engine PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")

# Link libraries
target_link_libraries(fluid_engine PRIVATE
    yaml-cpp::yaml-cpp
    glfw
    OpenGL::GL
    imgui
    onnxruntime
    Threads::Threads
)