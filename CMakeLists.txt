cmake_minimum_required(VERSION 3.10)

project(linear_algebra)

find_package(OpenMP REQUIRED)

set(SOURCES
    main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)