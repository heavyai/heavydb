set(logger_source_files
  Logger.cpp
)

add_library(Logger ${logger_source_files})
target_link_libraries(Logger ${Boost_LIBRARIES})
