add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift
    OUTPUT
        ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift)

add_library(calciteserver_thrift
    ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp
)
target_link_libraries(calciteserver_thrift ${Thrift_LIBRARIES})

add_library(Calcite Calcite.cpp Calcite.h)

target_link_libraries(Calcite Catalog calciteserver_thrift ${JAVA_JVM_LIBRARY})
