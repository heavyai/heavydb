add_library(Planner Planner.cpp Planner.h)

target_link_libraries(Planner Parser Analyzer)
