project(rlcpp)
add_definitions(-std=c++11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
    model/episode.cpp
    model/tablemodel.cpp
    learning/qlearning.cpp
    learning/advantagelearning.cpp
    learning/softmaxlearning.cpp
    world/abstractworld.cpp
    world/gridworld.cpp
    main.cpp
)

add_executable(rlcpp ${SOURCES})
