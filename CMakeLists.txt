cmake_minimum_required(VERSION 2.6)

project(rlcpp)
add_definitions(-std=c++11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3 REQUIRED)
pkg_check_modules(ROSCPP roscpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS})

set(SOURCES
    functionapproximators/gaussianmixture.cpp
    functionapproximators/fusionart.cpp
    model/episode.cpp
    model/tablemodel.cpp
    model/gaussianmixturemodel.cpp
    model/fusionartmodel.cpp
    model/nnetmodel.cpp
    model/perceptronmodel.cpp
    model/recurrentnnetmodel.cpp
    model/stackedgrumodel.cpp
    model/parallelgrumodel.cpp
    model/stackedlstmmodel.cpp
    learning/abstracttdlearning.cpp
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
    world/scaleworld.cpp
    deviceworld/deviceworld.cpp
    deviceworld/deviceintegratorworld.cpp
    modelbased/modelworld.cpp
    modelbased/dynamodel.cpp
    modelbased/texploremodel.cpp
    main.cpp
)

if (ROSCPP_FOUND)
    set(SOURCES ${SOURCES} world/rosworld.cpp)
    add_definitions(-DROSCPP_FOUND)
endif(ROSCPP_FOUND)

add_executable(rlcpp ${SOURCES})
target_link_libraries(rlcpp
    nnetcpp
    pthread
    ${ROSCPP_LDFLAGS}
)
