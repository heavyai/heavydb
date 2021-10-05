find_package(Boost COMPONENTS program_options filesystem regex system REQUIRED QUIET)

include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/linenoise)

add_executable(omnisql omnisql.cpp)
target_compile_definitions(omnisql PRIVATE  WIN32_LEAN_AND_MEAN) 

if(MSVC AND NOT ENABLE_FOLLY)
  # Not sure how this builds on Linux.
  # a Folly seems to be a maditory dependency.
  message(WARNING "omnisql build will fail on win32; requires folly to be enabled")
endif()

target_link_libraries(omnisql mapd_thrift Logger Shared ThriftClient linenoise ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${PROFILER_LIBS} ${Thrift_LIBRARIES} ${Folly_LIBRARIES})
if(ENABLE_KRB5)
  target_link_libraries(omnisql krb5_gss)
endif()

install(TARGETS omnisql DESTINATION bin COMPONENT "exe")
