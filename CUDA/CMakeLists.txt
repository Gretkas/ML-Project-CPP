cmake_minimum_required(VERSION 3.1)

project(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -Wall -Wextra")

add_executable(CUDA main.cpp)
