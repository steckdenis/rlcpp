cmake_minimum_required(VERSION 2.6)

project(rlcpp)
add_definitions(-std=c++11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3 REQUIRED)
pkg_check_modules(ROSCPP roscpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS})

set(SOURCES
    functionapproximators/gaussianmixture.cpp
    model/episode.cpp
    model/tablemodel.cpp
    model/gaussianmixturemodel.cpp
    model/nnetmodel.cpp
    model/perceptronmodel.cpp
    model/recurrentnnetmodel.cpp
    model/stackedgrumodel.cpp
    model/stackedlstmmodel.cpp
    learning/qlearning.cpp
    learning/advantagelearning.cpp
    learning/softmaxlearning.cpp
    learning/adaptivesoftmaxlearning.cpp
    learning/egreedylearning.cpp
    world/abstractworld.cpp
    world/tmazeworld.cpp
    world/gridworld.cpp
    world/polargridworld.cpp
    world/postprocessworld.cpp
    world/oneofnworld.cpp
    world/scaleworld.cpp
    main.cpp
)

if (ROSCPP_FOUND)
    set(SOURCES ${SOURCES} world/rosworld.cpp)
    add_definitions(-DROSCPP_FOUND)
endif(ROSCPP_FOUND)

add_executable(rlcpp ${SOURCES})
target_link_libraries(rlcpp
    nnetcpp
    ${ROSCPP_LDFLAGS}
)
