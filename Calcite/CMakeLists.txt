find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR}
${StringDictionary_BINARY_DIR}/gen-cpp)

add_library(Calcite CalciteJNI.cpp)

target_link_libraries(Calcite ${JAVA_JVM_LIBRARY})
