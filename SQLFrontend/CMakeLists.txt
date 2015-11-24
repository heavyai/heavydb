cmake_minimum_required(VERSION 2.8)
#set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3 -g")
FIND_PACKAGE( Boost COMPONENTS program_options filesystem system REQUIRED )
ADD_EXECUTABLE (mapdql mapdql.cpp linenoise.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)

set_source_files_properties(${mapd_thrift_gen_files} GENERATED)

target_link_libraries (mapdql Shared ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${Thrift_LIBRARIES} ${CMAKE_DL_LIBS})

install(TARGETS mapdql DESTINATION bin)
