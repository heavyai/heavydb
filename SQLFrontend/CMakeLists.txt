cmake_minimum_required(VERSION 2.8)
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3 -g")
FIND_PACKAGE( Boost COMPONENTS program_options REQUIRED )
ADD_EXECUTABLE (mapdql mapdql.cpp linenoise.cpp ${CMAKE_CURRENT_BINARY_DIR}/../gen-cpp/MapD.cpp ${CMAKE_CURRENT_BINARY_DIR}/../gen-cpp/mapd_constants.cpp ${CMAKE_CURRENT_BINARY_DIR}/../gen-cpp/mapd_types.cpp)

add_custom_command(
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../mapd.thrift
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/../gen-cpp/MapD.cpp
    COMMAND thrift
    ARGS -gen cpp -o ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../mapd.thrift)

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/../gen-cpp/MapD.cpp GENERATED)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/../gen-cpp/mapd_constants.cpp GENERATED)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/../gen-cpp/mapd_types.cpp GENERATED)

target_link_libraries (mapdql ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "-lthrift")
