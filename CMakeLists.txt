cmake_minimum_required(VERSION 2.8)

PROJECT(ImageResize)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Boost COMPONENTS filesystem regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})


find_package(OpenCV REQUIRED)	
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIR_SRCS)

add_executable(ImageResize ${DIR_SRCS})
target_link_libraries(ImageResize -lboost_system ${OpenCV_LIBS} ${Boost_LIBRARIES})



