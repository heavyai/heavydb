cmake_minimum_required(VERSION 2.8)

SET (analyzer_source_files
    Calcite.cpp
    Calcite.h
    ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp
)

include_directories("/usr/local/include")

ADD_LIBRARY (Calcite ${analyzer_source_files})

TARGET_LINK_LIBRARIES (Calcite Catalog)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/java/calciteserver.thrift
    OUTPUT ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/java/calciteserver.thrift)
