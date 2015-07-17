project(rlcpp)
add_definitions(-std=c++11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS})

set(SOURCES
    functionapproximators/gaussianmixture.cpp
    model/episode.cpp
    model/tablemodel.cpp
    model/gaussianmixturemodel.cpp
    model/nnetmodel.cpp
    learning/qlearning.cpp
    learning/advantagelearning.cpp
    learning/softmaxlearning.cpp
    learning/egreedylearning.cpp
    world/abstractworld.cpp
    world/gridworld.cpp
    world/oneofnworld.cpp
    main.cpp
)

add_executable(rlcpp ${SOURCES})
target_link_libraries(rlcpp nnetcpp)
