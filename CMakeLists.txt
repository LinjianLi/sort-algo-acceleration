#cmake_minimum_required(VERSION 3.5)
project(sort_algo_acceleration C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -mavx -mavx2")

add_executable(sort main.c)
add_executable(rdx_sort radix_sort.c)
