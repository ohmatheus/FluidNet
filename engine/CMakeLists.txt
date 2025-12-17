cmake_minimum_required(VERSION 3.20)
project(FluidNetEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(fluid_engine ${SOURCES})

target_include_directories(fluid_engine PRIVATE include)