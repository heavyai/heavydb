cmake_minimum_required(VERSION 2.8)
#set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3 -g")
FIND_PACKAGE( Boost COMPONENTS program_options filesystem system REQUIRED )
ADD_EXECUTABLE (mapdql mapdql.cpp linenoise.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
    OUTPUT ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/mapd.thrift)

set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp GENERATED)

target_link_libraries (mapdql Shared ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${Thrift_LIBRARIES} ${CMAKE_DL_LIBS} "-lncurses")

install(TARGETS mapdql DESTINATION bin)
