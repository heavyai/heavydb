cmake_minimum_required(VERSION 2.8)
set(RUNTIME_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

find_package(Boost COMPONENTS system program_options regex REQUIRED QUIET)
find_package(Thrift REQUIRED)

add_executable(StreamInsertSimple StreamInsertSimple.cpp)
add_executable(DataGen DataGen.cpp)
add_executable(StreamInsert StreamInsert.cpp)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift ${CMAKE_SOURCE_DIR}/completion_hints.thrift
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/MapD.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/mapd_constants.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/mapd_types.cpp
	${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/completion_hints_constants.cpp
	${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/completion_hints_types.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -r -I ${CMAKE_SOURCE_DIR} -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/mapd.thrift)
add_library(mapd_sample_thrift
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/MapD.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/mapd_constants.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/mapd_types.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/completion_hints_constants.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/completion_hints_types.cpp)
target_link_libraries(mapd_sample_thrift ${Thrift_LIBRARIES})

target_link_libraries(StreamInsertSimple mapd_sample_thrift ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
target_link_libraries(StreamInsert mapd_sample_thrift ${CMAKE_DL_LIBS} ${Boost_LIBRARIES})
target_link_libraries(DataGen mapd_sample_thrift ${CMAKE_DL_LIBS})

install(TARGETS StreamInsertSimple StreamInsert DataGen DESTINATION SampleCode)
install(DIRECTORY . DESTINATION SampleCode)
install(
    FILES
        ${CMAKE_SOURCE_DIR}/mapd.thrift
        ${CMAKE_SOURCE_DIR}/cmake/Modules/FindThrift.cmake
    DESTINATION SampleCode)
