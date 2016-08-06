cmake_minimum_required(VERSION 2.8.3)
project(pcl_utils)

find_package(EIGEN3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE libfiles src/*.cpp)
message(STATUS "libfiles: " ${libfiles})

add_library(pcl_utils_lib ${libfiles})
target_link_libraries(pcl_utils_lib 
    ${EIGEN3_LIBRARIES}
    ${PCL_LIBRARIES}
)

