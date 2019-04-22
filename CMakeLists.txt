cmake_minimum_required(VERSION 3.13)
project(sort_algo_acceleration C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

add_executable(sort main.c)