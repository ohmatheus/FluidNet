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

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(fluid_engine ${SOURCES})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/EngineConfig.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/EngineConfig.hpp
)

target_include_directories(fluid_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

# Link libraries
target_link_libraries(fluid_engine PRIVATE
    yaml-cpp::yaml-cpp
    glfw
    OpenGL::GL
    imgui
)