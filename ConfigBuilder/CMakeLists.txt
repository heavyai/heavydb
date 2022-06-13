add_library(ConfigBuilder ConfigBuilder.cpp)

target_link_libraries(ConfigBuilder Logger ${Boost_LIBRARIES})
