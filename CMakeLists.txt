cmake_minimum_required(VERSION 3.16)
project(parallel)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc -fopenmp -std=gnu++2a -m64")

add_executable(parallel main.cpp)
add_executable(aphi fib.cpp)
add_executable(avg average.cpp)
add_executable(srand srandom.cpp)
add_executable(rand random.cpp)
target_link_libraries(parallel PUBLIC OpenMP::OpenMP_CXX pthread)
