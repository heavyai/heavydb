set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(TEST_BASE_PATH "./tmp")
add_definitions("-DBASE_PATH=\"${TEST_BASE_PATH}\"")

add_executable(ExecuteTest ExecuteTest.cpp QueryRunner.cpp)
add_executable(RunQueryLoop RunQueryLoop.cpp QueryRunner.cpp)
add_executable(StringDictionaryTest StringDictionaryTest.cpp)
add_executable(PlanTest PlanTest.cpp)
add_executable(ProfileTest ProfileTest.cpp)

if(ENABLE_CUDA)
  set(MAPD_DEFINITIONS -DHAVE_CUDA)
  add_custom_command(
    DEPENDS ProfileUtils.cu ProfileTest.h
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -O3
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -arch sm_30
        -std=c++11
        -shared
        ${CMAKE_CURRENT_SOURCE_DIR}/ProfileUtils.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so
    )
  add_custom_target(ProfileUtils DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so)
  add_dependencies(ProfileTest ProfileUtils)
  set(PROF_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so)
endif()

add_executable(ResultSetTest ResultSetTest.cpp ResultSetTestUtils.cpp)
add_executable(ResultSetBaselineRadixSortTest ResultSetBaselineRadixSortTest.cpp ResultSetTestUtils.cpp)
add_executable(UtilTest UtilTest.cpp)
add_executable(StorageTest StorageTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
add_executable(StoragePerfTest StoragePerfTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
add_executable(ImportTest ImportTest.cpp QueryRunner.cpp)
add_executable(TopKTest TopKTest.cpp QueryRunner.cpp)
add_executable(TokenCompletionHintsTest TokenCompletionHintsTest.cpp)
add_executable(MapDQLCommandTest MapDQLCommandTest.cpp)

target_link_libraries(PlanTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(ProfileTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${PROF_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(UtilTest Utils gtest ${Boost_LIBRARIES})
target_link_libraries(StorageTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(StoragePerfTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
set(EXECUTE_TEST_LIBS gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
list(APPEND EXECUTE_TEST_LIBS Calcite)
target_link_libraries(ExecuteTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ResultSetTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ResultSetBaselineRadixSortTest ${EXECUTE_TEST_LIBS})
target_link_libraries(RunQueryLoop ${EXECUTE_TEST_LIBS})
target_link_libraries(StringDictionaryTest StringDictionary gtest ${Boost_LIBRARIES})
target_link_libraries(ImportTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(TopKTest ${EXECUTE_TEST_LIBS})
target_link_libraries(TokenCompletionHintsTest token_completion_hints gtest mapd_thrift ${Boost_LIBRARIES})
target_link_libraries(MapDQLCommandTest gtest ${EXECUTE_TEST_LIBS} ${Boost_LIBRARIES})

set(TEST_ARGS "--gtest_output=xml:../")
add_test(PlanTest PlanTest ${TEST_ARGS})
add_test(ImportTest ImportTest ${TEST_ARGS})
add_test(UtilTest UtilTest ${TEST_ARGS})
add_test(ExecuteTest ExecuteTest ${TEST_ARGS})
add_test(ResultSetTest ResultSetTest ${TEST_ARGS})
add_test(ResultSetBaselineRadixSortTest ResultSetBaselineRadixSortTest ${TEST_ARGS})
add_test(RunQueryLoop RunQueryLoop ${TEST_ARGS})
add_test(StringDictionaryTest StringDictionaryTest ${TEST_ARGS})
add_test(StorageTest StorageTest ${TEST_ARGS})
add_test(StoragePerfTest StoragePerfTest ${TEST_ARGS})
add_test(TopKTest TopKTest ${TEST_ARGS})
add_test(TokenCompletionHintsTest TokenCompletionHintsTest ${TEST_ARGS})
add_test(MapDQLCommandTest MapDQLCommandTest ${TEST_ARGS})

# parse s3 credentials
file(READ aws/s3client.conf S3CLIENT_CONF)
if("${S3CLIENT_CONF}" MATCHES "AWS_ACCESS_KEY_ID=([^\n]+)")
  set(AWS_ACCESS_KEY_ID "${CMAKE_MATCH_1}")
endif()
if("${S3CLIENT_CONF}" MATCHES "AWS_SECRET_ACCESS_KEY=([^\n]+)")
  set(AWS_SECRET_ACCESS_KEY "${CMAKE_MATCH_1}")
endif()
if("${S3CLIENT_CONF}" MATCHES "AWS_REGION=([^\n]+)")
  set(AWS_REGION "${CMAKE_MATCH_1}")
endif()

set(SANITY_TESTS
  PlanTest
  ExecuteTest
  ResultSetTest
  ResultSetBaselineRadixSortTest
  StorageTest
  ImportTest
  TopKTest
  TokenCompletionHintsTest
  MapDQLCommandTest
)
set_tests_properties(${SANITY_TESTS} PROPERTIES LABELS "sanity")

add_custom_target(sanity_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND env AWS_REGION=${AWS_REGION} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} ${CMAKE_CTEST_COMMAND} --verbose --label-regex sanity
    DEPENDS ${SANITY_TESTS})

add_custom_target(all_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND env AWS_REGION=${AWS_REGION} AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS ${SANITY_TESTS} ProfileTest UtilTest RunQueryLoop StringDictionaryTest StoragePerfTest)

add_custom_target(storage_perf_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --tests-regex "\"(StoragePerfTest)\""
    DEPENDS StoragePerfTest)

add_custom_target(topk_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --tests-regex "\"(TopKTest)\""
    DEPENDS TopKTest)
