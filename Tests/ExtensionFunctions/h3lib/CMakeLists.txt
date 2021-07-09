set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions("-DBASE_PATH=\"${TEST_BASE_PATH}\"")

add_executable(H3LibExtFuncTest H3LibExtFuncTest.cpp)
target_link_libraries(H3LibExtFuncTest ${EXECUTE_TEST_LIBS})

add_test(H3LibExtFuncTest H3LibExtFuncTest ${TEST_ARGS})

# Creating some symlinks to make hardcoded paths to bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar
# and QueryEngine/RuntimeFunctions.bc
# The directories of each of the above files are supposed to be one level above the test executable, but
# since we are burying this particular test a few levels deeper, these path lookups fail
file(CREATE_LINK ${CMAKE_BINARY_DIR}/Tests/QueryEngine ${EXECUTABLE_OUTPUT_PATH}/../QueryEngine SYMBOLIC)
file(CREATE_LINK ${CMAKE_BINARY_DIR}/bin ${EXECUTABLE_OUTPUT_PATH}/../bin SYMBOLIC)
