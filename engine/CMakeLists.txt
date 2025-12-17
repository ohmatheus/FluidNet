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

# Configure yaml-cpp build options before making it available
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib")
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "Build yaml-cpp as static library")

FetchContent_MakeAvailable(yaml-cpp)

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
)