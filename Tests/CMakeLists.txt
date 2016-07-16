cmake_minimum_required(VERSION 2.8)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(TEST_BASE_PATH "./tmp")
add_definitions("-DBASE_PATH=\"${TEST_BASE_PATH}\"")

add_executable(ExecuteTest ExecuteTest.cpp QueryRunner.cpp)
add_executable(RunQueryLoop RunQueryLoop.cpp QueryRunner.cpp)
add_executable(StringDictionaryTest StringDictionaryTest.cpp)
add_executable(PlanTest PlanTest.cpp)
add_executable(ResultSetTest ResultSetTest.cpp)
add_executable(UtilTest UtilTest.cpp)
add_executable(StorageTest StorageTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
add_executable(StoragePerfTest StoragePerfTest.cpp PopulateTableRandom.cpp ScanTable.cpp)

target_link_libraries(PlanTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} "-lz -lncurses")
target_link_libraries(UtilTest Utils gtest)
target_link_libraries(StorageTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} "-lz -lncurses")
target_link_libraries(StoragePerfTest gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} "-lz -lncurses")
set(EXECUTE_TEST_LIBS gtest ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} "-lz -lncurses")
if(ENABLE_CALCITE)
  list(APPEND EXECUTE_TEST_LIBS Calcite)
endif()
target_link_libraries(ExecuteTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ResultSetTest ${EXECUTE_TEST_LIBS})
target_link_libraries(RunQueryLoop ${EXECUTE_TEST_LIBS})
target_link_libraries(StringDictionaryTest StringDictionary gtest ${Boost_LIBRARIES})

set(TEST_ARGS "--gtest_output=xml:../")
add_test(PlanTest PlanTest ${TEST_ARGS})
add_test(UtilTest UtilTest ${TEST_ARGS})
add_test(ExecuteTest ExecuteTest ${TEST_ARGS})
add_test(RunQueryLoop RunQueryLoop ${TEST_ARGS})
add_test(StringDictionaryTest StringDictionaryTest ${TEST_ARGS})
add_test(StorageTest StorageTest ${TEST_ARGS})
add_test(StoragePerfTest StoragePerfTest ${TEST_ARGS})

add_custom_target(sanity_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --tests-regex "\"(PlanTest|ExecuteTest|StorageTest)\""
    DEPENDS PlanTest ExecuteTest StorageTest)

add_custom_target(all_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS PlanTest UtilTest ExecuteTest RunQueryLoop StringDictionaryTest StorageTest StoragePerfTest)
