find_package(Boost COMPONENTS program_options filesystem regex system REQUIRED QUIET)

include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/linenoise)

add_executable(heavysql heavysql.cpp)
target_compile_definitions(heavysql PRIVATE  WIN32_LEAN_AND_MEAN)

target_link_libraries(heavysql mapd_thrift Logger Shared ThriftClient linenoise ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${PROFILER_LIBS} ${Thrift_LIBRARIES})
if(ENABLE_KRB5)
  target_link_libraries(heavysql krb5_gss)
endif()

install(TARGETS heavysql DESTINATION bin COMPONENT "exe")
