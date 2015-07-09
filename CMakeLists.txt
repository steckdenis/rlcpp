project(rlcpp)
add_definitions(-std=c++11)

set(SOURCES
    model/episode.cpp
    model/tablemodel.cpp
    main.cpp
)

add_executable(rlcpp ${SOURCES})
