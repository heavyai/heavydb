cmake_minimum_required(VERSION 2.8)
#set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3 -g")
find_package(Boost COMPONENTS program_options filesystem system REQUIRED)
add_executable(mapdql mapdql.cpp linenoise.cpp)

target_link_libraries(mapdql mapd_thrift Shared ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS})

install(TARGETS mapdql DESTINATION bin)
