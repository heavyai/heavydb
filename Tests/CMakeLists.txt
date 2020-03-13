set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(TEST_BASE_PATH "./tmp")
add_definitions("-DBASE_PATH=\"${TEST_BASE_PATH}\"")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Udf/udf_sample.cpp DESTINATION ${CMAKE_BINARY_DIR}/Tests/Udf)

add_executable(CodeGeneratorTest CodeGeneratorTest.cpp)
add_executable(ArrowIpcIntegrationTest ArrowIpcIntegrationTest.cpp)
add_executable(ExecuteTest ExecuteTest.cpp ClusterTester.cpp)
add_executable(RunQueryLoop RunQueryLoop.cpp)
add_executable(StringDictionaryTest StringDictionaryTest.cpp)
add_executable(StringTransformTest StringTransformTest.cpp)
add_executable(StringFunctionsTest StringFunctionsTest.cpp)
add_executable(PlanTest PlanTest.cpp)
add_executable(ProfileTest ProfileTest.cpp)
add_executable(ExperimentalTest ExperimentalTest.cpp)
add_executable(ForeignServerDdlTest ForeignServerDdlTest.cpp)
add_executable(CatalogMigrationTest CatalogMigrationTest.cpp)
add_executable(CreateAndDropTableDdlTest CreateAndDropTableDdlTest.cpp)

if(ENABLE_CUDA)
  set(MAPD_DEFINITIONS -DHAVE_CUDA)
  if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
    list(APPEND MAPD_DEFINITIONS -DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
  endif()
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
        -std=c++14
        -shared
        ${CMAKE_CURRENT_SOURCE_DIR}/ProfileUtils.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so
    )
  add_custom_target(ProfileUtils DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so)
  add_dependencies(ProfileTest ProfileUtils)
  set(PROF_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so)
endif()

add_executable(DumpRestoreTest DumpRestoreTest.cpp)
add_executable(ResultSetTest ResultSetTest.cpp ResultSetTestUtils.cpp)
add_executable(FromTableReorderingTest FromTableReorderingTest.cpp)
add_executable(ResultSetBaselineRadixSortTest ResultSetBaselineRadixSortTest.cpp ResultSetTestUtils.cpp)
add_executable(UtilTest UtilTest.cpp)
add_executable(StorageTest StorageTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
add_executable(StoragePerfTest StoragePerfTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
add_executable(ImportTest ImportTest.cpp)
add_executable(ArrowCsvForeignStorageTest ArrowCsvForeignStorageTest.cpp)
add_executable(AlterColumnTest AlterColumnTest.cpp)
add_executable(UpdelStorageTest UpdelStorageTest.cpp)
add_executable(ComputeMetadataTest ComputeMetadataTest.cpp)
add_executable(BumpAllocatorTest BumpAllocatorTest.cpp)
add_executable(SpecialCharsTest SpecialCharsTest.cpp)
add_executable(TableFunctionsTest TableFunctionsTest.cpp)
add_executable(ArrayTest ArrayTest.cpp)
add_executable(TopKTest TopKTest.cpp)
add_executable(TokenCompletionHintsTest TokenCompletionHintsTest.cpp)
add_executable(OmniSQLCommandTest OmniSQLCommandTest.cpp)
add_executable(OmniSQLUtilitiesTest OmniSQLUtilitiesTest.cpp)
add_executable(DBObjectPrivilegesTest DBObjectPrivilegesTest.cpp)
add_executable(GeoTypesTest Shared/GeoTypesTest.cpp)
add_executable(CtasUpdateTest CtasUpdateTest.cpp)
add_executable(CorrelatedSubqueryTest CorrelatedSubqueryTest.cpp)
add_executable(CtasIntegrationTest CtasIntegrationTest.cpp)
add_executable(DateTimeUtilsTest Shared/DateTimeUtilsTest.cpp)
add_executable(UpdateMetadataTest UpdateMetadataTest.cpp)
add_executable(CalciteOptimizeTest CalciteOptimizeTest.cpp)
add_executable(JoinHashTableTest JoinHashTableTest.cpp)
add_executable(ColumnarResultsTest ColumnarResultsTest.cpp ResultSetTestUtils.cpp)
add_executable(CommandLineTest CommandLineTest.cpp)
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  add_executable(UdfTest UdfTest.cpp)
endif()

target_link_libraries(ProfileTest gtest Shared Calcite QueryEngine ${MAPD_RENDERING_LIBRARIES} CsvImport QueryRunner QueryState Parser ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${PROF_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(ResultSetTest gtest QueryEngine ${MAPD_RENDERING_LIBRARIES} CsvImport QueryRunner QueryState Parser DataMgr Chunk Shared ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(ColumnarResultsTest gtest QueryEngine ${MAPD_RENDERING_LIBRARIES} CsvImport QueryRunner QueryState Parser DataMgr Chunk Shared ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(FromTableReorderingTest gtest QueryEngine ${MAPD_RENDERING_LIBRARIES} CsvImport QueryRunner QueryState Parser DataMgr Chunk ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(ResultSetBaselineRadixSortTest gtest QueryEngine ${MAPD_RENDERING_LIBRARIES} CsvImport QueryRunner QueryState Parser DataMgr Chunk Shared ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(UtilTest Utils gtest Shared ${Boost_LIBRARIES})
target_link_libraries(StringDictionaryTest StringDictionary gtest Shared ${Boost_LIBRARIES})
target_link_libraries(StringTransformTest Shared gtest ${Boost_LIBRARIES})
target_link_libraries(StringFunctionsTest gtest QueryRunner ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${LLVM_LINKER_FLAGS}
    ${CURSES_LIBRARIES} ${LOCALE_LINK_FLAG})
target_link_libraries(TokenCompletionHintsTest token_completion_hints gtest mapd_thrift Shared ${Boost_LIBRARIES})
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")  # work around linker on centos
  set(EXECUTE_TEST_LIBS gtest QueryRunner ${MAPD_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES} ${Boost_LIBRARIES} ${MAPD_LIBRARIES})
else()
  set(EXECUTE_TEST_LIBS gtest QueryRunner ${MAPD_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES} ${Boost_LIBRARIES})
endif()
list(APPEND EXECUTE_TEST_LIBS Calcite)
list(APPEND EXECUTE_TEST_LIBS Calcite mapd_thrift ${PROFILER_LIBS})

target_link_libraries(DumpRestoreTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(CodeGeneratorTest ${EXECUTE_TEST_LIBS} CsvImport QueryRunner QueryState)
target_link_libraries(ArrowIpcIntegrationTest gtest QueryEngine Shared mapd_thrift ${Arrow_LIBRARIES})
target_link_libraries(ExecuteTest ${EXECUTE_TEST_LIBS})
target_link_libraries(RunQueryLoop ${EXECUTE_TEST_LIBS} bcrypt)
target_link_libraries(ImportTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(ArrowCsvForeignStorageTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(AlterColumnTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(PlanTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(UpdelStorageTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(StorageTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(ComputeMetadataTest ${EXECUTE_TEST_LIBS})
target_link_libraries(TableFunctionsTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ArrayTest ${EXECUTE_TEST_LIBS})
target_link_libraries(BumpAllocatorTest ${EXECUTE_TEST_LIBS})
target_link_libraries(SpecialCharsTest ${EXECUTE_TEST_LIBS})
target_link_libraries(UpdateMetadataTest ${EXECUTE_TEST_LIBS})
target_link_libraries(StoragePerfTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(TopKTest ${EXECUTE_TEST_LIBS})
target_link_libraries(OmniSQLCommandTest gtest ${Boost_LIBRARIES} mapd_thrift)
if(ENABLE_KRB5)
	target_link_libraries(OmniSQLCommandTest krb5_gss Shared)
endif()
target_link_libraries(OmniSQLUtilitiesTest gtest ${Boost_LIBRARIES})
target_link_libraries(ExperimentalTest gtest Shared)
target_link_libraries(DBObjectPrivilegesTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(GeoTypesTest gtest Shared ${GDAL_LIBRARIES} ${GDALExtra_LIBRARIES} ${LibAwsS3_SUPPORT_LIBRARIES} ${CMAKE_DL_LIBS})
target_link_libraries(CtasUpdateTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(CorrelatedSubqueryTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(CtasIntegrationTest gtest Shared mapd_thrift ThriftClient ${LLVM_LINKER_FLAGS})
target_link_libraries(DateTimeUtilsTest gtest Shared ${LLVM_LINKER_FLAGS})
target_link_libraries(CalciteOptimizeTest gtest ${EXECUTE_TEST_LIBS} ${Boost_LIBRARIES})
target_link_libraries(JoinHashTableTest gtest ${EXECUTE_TEST_LIBS} CsvImport QueryRunner)
target_link_libraries(CommandLineTest gtest Shared ${Boost_LIBRARIES})
target_link_libraries(ForeignServerDdlTest gtest thrift_handler ${MAPD_LIBRARIES} ${LLVM_LINKER_FLAGS}
    ${CURSES_LIBRARIES})
target_link_libraries(CatalogMigrationTest gtest thrift_handler ${MAPD_LIBRARIES} ${RdKafka_LIBRARIES} ${MAPD_LIBRARIES}
    ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES})
target_link_libraries(CreateAndDropTableDdlTest gtest thrift_handler ${MAPD_LIBRARIES} ${LLVM_LINKER_FLAGS}
    ${CURSES_LIBRARIES})

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  target_link_libraries(UdfTest gtest ${EXECUTE_TEST_LIBS} CsvImport QueryRunner)
endif()

set(TEST_ARGS "--gtest_output=xml:../")
add_test(DumpRestoreTest DumpRestoreTest ${TEST_ARGS})
add_test(PlanTest PlanTest ${TEST_ARGS})
add_test(UpdelStorageTest UpdelStorageTest ${TEST_ARGS})
add_test(ImportTest ImportTest ${TEST_ARGS})
add_test(ArrowCsvForeignStorageTest ArrowCsvForeignStorageTest ${TEST_ARGS})
add_test(AlterColumnTest AlterColumnTest ${TEST_ARGS})
add_test(UtilTest UtilTest ${TEST_ARGS})
add_test(ExecuteTest ExecuteTest ${TEST_ARGS})
add_test(NAME ExecuteTestTemporaryTables COMMAND ExecuteTest ${TEST_ARGS} "--use-temporary-tables")
add_test(CodeGeneratorTest CodeGeneratorTest ${TEST_ARGS})
add_test(ArrowIpcIntegrationTest ArrowIpcIntegrationTest ${TEST_ARGS})
add_test(ResultSetTest ResultSetTest ${TEST_ARGS})
add_test(ColumnarResultsTest ColumnarResultsTest ${TEST_ARGS})
add_test(FromTableReorderingTest FromTableReorderingTest ${TEST_ARGS})
add_test(JoinHashTableTest JoinHashTableTest ${TEST_ARGS})
add_test(ResultSetBaselineRadixSortTest ResultSetBaselineRadixSortTest ${TEST_ARGS})
add_test(RunQueryLoop RunQueryLoop ${TEST_ARGS})
add_test(StringDictionaryTest StringDictionaryTest ${TEST_ARGS})
add_test(StringTransformTest StringTransformTest ${TEST_ARGS})
add_test(StringFunctionsTest StringFunctionsTest ${TEST_ARGS})
add_test(StorageTest StorageTest ${TEST_ARGS})
add_test(ComputeMetadataTest ComputeMetadataTest ${TEST_ARGS})
add_test(BumpAllocatorTest BumpAllocatorTest ${TEST_ARGS})
add_test(SpecialCharsTest SpecialCharsTest ${TEST_ARGS})
add_test(TableFunctionsTest TableFunctionsTest ${TEST_ARGS})
add_test(ArrayTest ArrayTest ${TEST_ARGS})
add_test(StoragePerfTest StoragePerfTest ${TEST_ARGS})
add_test(TopKTest TopKTest ${TEST_ARGS})
add_test(TokenCompletionHintsTest TokenCompletionHintsTest ${TEST_ARGS})
add_test(OmniSQLCommandTest OmniSQLCommandTest ${TEST_ARGS})
add_test(OmniSQLUtilitiesTest OmniSQLUtilitiesTest ${TEST_ARGS})
add_test(DBObjectPrivilegesTest DBObjectPrivilegesTest ${TEST_ARGS})
add_test(GeoTypesTest GeoTypesTest ${TEST_ARGS})
add_test(CtasUpdateTest CtasUpdateTest ${TEST_ARGS})
add_test(CorrelatedSubqueryTest CorrelatedSubqueryTest ${TEST_ARGS})
add_test(DateTimeUtilsTest DateTimeUtilsTest ${TEST_ARGS})
add_test(UpdateMetadataTest UpdateMetadataTest ${TEST_ARGS})
add_test(CalciteOptimizeTest CalciteOptimizeTest ${TEST_ARGS})
add_test(JoinHashTableTest JoinHashTableTest ${TEST_ARGS})
add_test(CommandLineTest CommandLineTest ${TEST_ARGS})
add_test(ForeignServerDdlTest ForeignServerDdlTest ${TEST_ARGS})
add_test(CatalogMigrationTest CatalogMigrationTest ${TEST_ARGS})
add_test(CreateAndDropTableDdlTest CreateAndDropTableDdlTest ${TEST_ARGS})

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  add_test(UdfTest UdfTest ${TEST_ARGS})
endif()

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

set(TEST_PROGRAMS 
  DumpRestoreTest
  PlanTest
  ExecuteTest
  CodeGeneratorTest
  ResultSetTest
  ColumnarResultsTest
  FromTableReorderingTest
  ResultSetBaselineRadixSortTest
  StorageTest
  ImportTest
  ArrowCsvForeignStorageTest
  AlterColumnTest
  UpdelStorageTest
  ComputeMetadataTest
  BumpAllocatorTest
  SpecialCharsTest
  TableFunctionsTest
  ArrayTest
  TopKTest
  TokenCompletionHintsTest
  OmniSQLCommandTest
  OmniSQLUtilitiesTest
  DBObjectPrivilegesTest
  GeoTypesTest
  CtasUpdateTest
  CorrelatedSubqueryTest
  DateTimeUtilsTest
  UpdateMetadataTest
  CalciteOptimizeTest
  JoinHashTableTest
  StringFunctionsTest
  CommandLineTest
  ForeignServerDdlTest
  CatalogMigrationTest
  CreateAndDropTableDdlTest
)

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  list(APPEND TEST_PROGRAMS UdfTest)
endif()

set(SANITY_TESTS ${TEST_PROGRAMS})
list(APPEND SANITY_TESTS ExecuteTestTemporaryTables)

set_tests_properties(${SANITY_TESTS} PROPERTIES LABELS "sanity")

add_custom_target(sanity_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose --label-regex sanity
    DEPENDS ${TEST_PROGRAMS}
    USES_TERMINAL)

add_custom_target(all_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS ${TEST_PROGRAMS} ProfileTest UtilTest RunQueryLoop StringDictionaryTest StringTransformTest StoragePerfTest
    USES_TERMINAL)

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
