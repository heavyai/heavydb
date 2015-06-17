cmake_minimum_required(VERSION 2.8)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
FIND_PACKAGE( Boost COMPONENTS system program_options regex REQUIRED )
ADD_EXECUTABLE (StreamInsertSimple StreamInsertSimple.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)
ADD_EXECUTABLE (DataGen DataGen.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)
ADD_EXECUTABLE (StreamInsert StreamInsert.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
    OUTPUT ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp
    COMMAND thrift
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/mapd.thrift)

set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp GENERATED)

target_link_libraries (StreamInsertSimple ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "-lthrift")
target_link_libraries (StreamInsert ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} "-lthrift")
target_link_libraries (DataGen ${CMAKE_DL_LIBS} "-lthrift")
