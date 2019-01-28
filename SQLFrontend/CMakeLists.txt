find_package(Boost COMPONENTS program_options filesystem regex system REQUIRED QUIET)

add_executable(omnisql omnisql.cpp)

target_link_libraries(omnisql mapd_thrift Shared ThriftClient linenoise ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${PROFILER_LIBS} ${Thrift_LIBRARIES})

install(TARGETS omnisql DESTINATION bin)
