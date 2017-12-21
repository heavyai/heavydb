find_package(Boost COMPONENTS program_options filesystem regex system REQUIRED QUIET)
add_executable(mapdql mapdql.cpp)

target_link_libraries(mapdql mapd_thrift Shared linenoise ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS})

install(TARGETS mapdql DESTINATION bin)
