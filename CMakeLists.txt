cmake_minimum_required (VERSION 3.8)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3 -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -FATAL")

project(model-based_labelimg2xml)

find_package(OpenCV CONFIG REQUIRED)
    
add_executable (labelit
    "model.cpp"
    "tinyxml2.cpp"
    )

target_link_libraries(labelit PRIVATE 
    opencv_calib3d
    opencv_core
    opencv_dnn
    opencv_features2d
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
    opencv_ml
    opencv_video
    opencv_videoio
    )