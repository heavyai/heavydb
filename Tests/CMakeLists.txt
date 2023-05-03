##########

########## Settings.

##########

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(TEST_BASE_PATH "./tmp")
add_definitions("-DBASE_PATH=\"${TEST_BASE_PATH}\"")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Udf/udf_sample.cpp DESTINATION ${CMAKE_BINARY_DIR}/Tests/Udf)

if(ENABLE_CUDA)
  message(DEBUG "Tests CUDA_COMPILATION_ARCH: ${CUDA_COMPILATION_ARCH}")
  set(MAPD_DEFINITIONS -DHAVE_CUDA)
  if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
    list(APPEND MAPD_DEFINITIONS -DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
  endif()
endif()

find_package(fmt REQUIRED)
#set(EXECUTE_TEST_LIBS gtest mapd_thrift QueryRunner fmt::fmt ${MAPD_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} ${ZLIB_LIBRARIES})
# If ENABLE_AWS_S3=off AwsS3_CURL_SUPPORT will be empty
set(EXECUTE_TEST_LIBS gtest mapd_thrift QueryRunner fmt::fmt ${MAPD_LIBRARIES}
  ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${AwsS3_CURL_SUPPORT})

set(THRIFT_HANDLER_TEST_LIBRARIES thrift_handler ${EXECUTE_TEST_LIBS})

#parse s3 credentials
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

set(TEST_ARGS "--gtest_output=xml:../")

# Partition settings for the sanity_tests.
set(HEAVYAI_PARTITION 0 CACHE STRING "Partition number to build or zero to disable partitioning.")
set(HEAVYAI_MAX_PARTITION 4)
if(${HEAVYAI_PARTITION} LESS 0 OR ${HEAVYAI_PARTITION} GREATER ${HEAVYAI_MAX_PARTITION})
  message(FATAL_ERROR "invalid HEAVYAI_PARTITION ${HEAVYAI_PARTITION}, expected 1-${HEAVYAI_MAX_PARTITION}, or 0 to disable")
endif()
if(${HEAVYAI_PARTITION} GREATER_EQUAL 1 AND ${HEAVYAI_PARTITION} LESS_EQUAL ${HEAVYAI_MAX_PARTITION})
  message(STATUS "Building only sanity tests from partition: ${HEAVYAI_PARTITION} (of ${HEAVYAI_MAX_PARTITION})")
endif()

##########

########## Sanity tests (sanity_tests).

##########

if(${HEAVYAI_PARTITION} EQUAL 0 OR ${HEAVYAI_PARTITION} EQUAL 1)  # Partition 1

##########

add_executable(AlterSystemTest AlterSystemTest.cpp)
target_link_libraries(AlterSystemTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(AlterSystemTest AlterSystemTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS AlterSystemTest)

##########

add_executable(AlterTableDdlTest AlterTableDdlTest.cpp)
target_link_libraries(AlterTableDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(AlterTableDdlTest AlterTableDdlTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS AlterTableDdlTest)

##########

add_executable(ArrayTest ArrayTest.cpp)
target_link_libraries(ArrayTest ${EXECUTE_TEST_LIBS})
add_test(ArrayTest ArrayTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ArrayTest)

##########

add_executable(ArrowCsvForeignStorageTest ArrowCsvForeignStorageTest.cpp)
target_link_libraries(ArrowCsvForeignStorageTest gtest ${EXECUTE_TEST_LIBS})
add_test(ArrowCsvForeignStorageTest ArrowCsvForeignStorageTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ArrowCsvForeignStorageTest)

##########

add_executable(BufferMgrTest BufferMgrTest.cpp)
target_link_libraries(BufferMgrTest gtest DataMgr QueryEngine)
add_test(BufferMgrTest BufferMgrTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS BufferMgrTest)

##########

add_executable(BumpAllocatorTest BumpAllocatorTest.cpp)
target_link_libraries(BumpAllocatorTest ${EXECUTE_TEST_LIBS})
add_test(BumpAllocatorTest BumpAllocatorTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS BumpAllocatorTest)

##########

add_executable(CachedHashTableTest CachedHashTableTest.cpp)
target_link_libraries(CachedHashTableTest ${EXECUTE_TEST_LIBS})
add_test(CachedHashTableTest CachedHashTableTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CachedHashTableTest)

##########

add_executable(CachingFileMgrTest CachingFileMgrTest.cpp)
target_link_libraries(CachingFileMgrTest DataMgr ${EXECUTE_TEST_LIBS})
add_test(CachingFileMgrTest CachingFileMgrTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CachingFileMgrTest)

##########

add_executable(CalciteOptimizeTest CalciteOptimizeTest.cpp)
target_link_libraries(CalciteOptimizeTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(CalciteOptimizeTest CalciteOptimizeTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CalciteOptimizeTest)

##########

add_executable(CatalogMigrationTest CatalogMigrationTest.cpp)
target_link_libraries(CatalogMigrationTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(CatalogMigrationTest CatalogMigrationTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CatalogMigrationTest)

##########

add_executable(CodeGeneratorTest CodeGeneratorTest.cpp)
target_link_libraries(CodeGeneratorTest ${EXECUTE_TEST_LIBS} ${LibAwsS3_LIBRARIES})
add_test(CodeGeneratorTest CodeGeneratorTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CodeGeneratorTest)

##########

add_executable(ColumnarResultsTest ColumnarResultsTest.cpp ResultSetTestUtils.cpp)
target_link_libraries(ColumnarResultsTest ${EXECUTE_TEST_LIBS} ${LibAwsS3_LIBRARIES})
add_test(ColumnarResultsTest ColumnarResultsTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ColumnarResultsTest)

##########

add_executable(CommandLineTest CommandLineTest.cpp)
target_link_libraries(CommandLineTest gtest Logger Shared ${Boost_LIBRARIES})
add_test(CommandLineTest CommandLineTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CommandLineTest)

##########

add_executable(ComputeMetadataTest ComputeMetadataTest.cpp)
target_link_libraries(ComputeMetadataTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ComputeMetadataTest ComputeMetadataTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ComputeMetadataTest)

##########

add_executable(ConcurrencyTest ConcurrencyTest.cpp)
target_link_libraries(ConcurrencyTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ConcurrencyTest ConcurrencyTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ConcurrencyTest)

##########

add_executable(CorrelatedSubqueryTest CorrelatedSubqueryTest.cpp)
target_link_libraries(CorrelatedSubqueryTest ${EXECUTE_TEST_LIBS})
add_test(CorrelatedSubqueryTest CorrelatedSubqueryTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CorrelatedSubqueryTest)

##########

add_executable(CreateAndDropTableDdlTest CreateAndDropTableDdlTest.cpp)
target_link_libraries(CreateAndDropTableDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(CreateAndDropTableDdlTest CreateAndDropTableDdlTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CreateAndDropTableDdlTest)

##########

add_executable(CrossDatabaseQueryTest CrossDatabaseQueryTest.cpp)
target_link_libraries(CrossDatabaseQueryTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(CrossDatabaseQueryTest CrossDatabaseQueryTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS CrossDatabaseQueryTest)

##########

if(NOT MSVC)
  add_executable(CtasUpdateTest CtasUpdateTest.cpp)
  target_link_libraries(CtasUpdateTest ${THRIFT_HANDLER_TEST_LIBRARIES})
  add_test(CtasUpdateTest CtasUpdateTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS CtasUpdateTest)
endif()

##########

add_executable(DashboardAndCustomExpressionTest DashboardAndCustomExpressionTest.cpp)
target_link_libraries(DashboardAndCustomExpressionTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(DashboardAndCustomExpressionTest DashboardAndCustomExpressionTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS DashboardAndCustomExpressionTest)

##########

add_executable(DataMgrTest DataMgrTest.cpp)
target_link_libraries(DataMgrTest DataMgr ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(DataMgrTest DataMgrTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS DataMgrTest)

##########

add_executable(DataRecyclerTest DataRecyclerTest.cpp)
target_link_libraries(DataRecyclerTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(DataRecyclerTest DataRecyclerTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS DataRecyclerTest)

##########

add_executable(DateTimeUtilsTest Shared/DateTimeUtilsTest.cpp)
target_link_libraries(DateTimeUtilsTest gtest Logger Shared ${LLVM_LINKER_FLAGS})
add_test(DateTimeUtilsTest DateTimeUtilsTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS DateTimeUtilsTest)

##########

add_executable(DBObjectPrivilegesTest DBObjectPrivilegesTest.cpp)
target_link_libraries(DBObjectPrivilegesTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(DBObjectPrivilegesTest DBObjectPrivilegesTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS DBObjectPrivilegesTest)

##########

add_executable(DiskCacheQueryTest DiskCacheQueryTest.cpp)
target_link_libraries(DiskCacheQueryTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(DiskCacheQueryTest DiskCacheQueryTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS DiskCacheQueryTest)

##########

add_executable(DumpRestoreTest DumpRestoreTest.cpp)
target_link_libraries(DumpRestoreTest ${EXECUTE_TEST_LIBS})
add_test(DumpRestoreTest DumpRestoreTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS DumpRestoreTest)

##########

if(NOT ENABLE_CUDA AND ENABLE_DBE)
  add_executable(EmbeddedDatabaseTest EmbeddedDatabaseTest.cpp)
  target_link_libraries(EmbeddedDatabaseTest gtest DBEngine)
  add_test(EmbeddedDatabaseTest EmbeddedDatabaseTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS EmbeddedDatabaseTest)
endif()

##########

add_executable(EncoderTest EncoderTest.cpp)
target_link_libraries(EncoderTest gtest DataMgr Shared Logger)
add_test(EncoderTest EncoderTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS EncoderTest)

##########

endif()  # Partition 1
if(${HEAVYAI_PARTITION} EQUAL 0 OR ${HEAVYAI_PARTITION} EQUAL 2)  # Partition 2

##########

add_executable(ExecuteTest ExecuteTest.cpp ClusterTester.cpp)
target_link_libraries(ExecuteTest ${EXECUTE_TEST_LIBS})
add_test(ExecuteTest ExecuteTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ExecuteTest)
add_test(NAME ExecuteTestTemporaryTables COMMAND ExecuteTest ${TEST_ARGS} "--use-temporary-tables")
add_test(NAME ExecuteTestExecutorResourceMgr COMMAND ExecuteTest ${TEST_ARGS} "--enable-executor-resource-mgr")
list(APPEND INSANITY_TEST_PROGRAMS ExecuteTest)
list(APPEND INSANITY_TESTS ExecuteTestTemporaryTables)
list(APPEND INSANITY_TESTS ExecuteTestExecutorResourceMgr)

##########

endif()  # Partition 2
if(${HEAVYAI_PARTITION} EQUAL 0 OR ${HEAVYAI_PARTITION} EQUAL 3)  # Partition 3

##########

add_executable(ExecutorResourceMgrTest ExecutorResourceMgrTest.cpp)
target_link_libraries(ExecutorResourceMgrTest ${EXECUTE_TEST_LIBS})
add_test(ExecutorResourceMgrTest ExecutorResourceMgrTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ExecutorResourceMgrTest)

##########

add_executable(ExplainTest ExplainTest.cpp)
target_link_libraries(ExplainTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ExplainTest ExplainTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ExplainTest)

##########

add_executable(FileMgrTest FileMgrTest.cpp)
target_link_libraries(FileMgrTest DataMgr ${EXECUTE_TEST_LIBS})
add_test(FileMgrTest FileMgrTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS FileMgrTest)

##########

add_executable(FilePathWhitelistTest FilePathWhitelistTest.cpp)
target_link_libraries(FilePathWhitelistTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(FilePathWhitelistTest FilePathWhitelistTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS FilePathWhitelistTest)

##########

add_executable(ForeignServerDdlTest ForeignServerDdlTest.cpp)
target_link_libraries(ForeignServerDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ForeignServerDdlTest ForeignServerDdlTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ForeignServerDdlTest)

##########

add_executable(ForeignStorageCacheTest ForeignStorageCacheTest.cpp)
target_link_libraries(ForeignStorageCacheTest gtest ${MAPD_LIBRARIES} ${AwsS3_CURL_SUPPORT})
add_test(ForeignStorageCacheTest ForeignStorageCacheTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ForeignStorageCacheTest)

##########

add_executable(ForeignTableDmlTest ForeignTableDmlTest.cpp)
target_link_libraries(ForeignTableDmlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ForeignTableDmlTest ForeignTableDmlTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ForeignTableDmlTest)

##########

add_executable(FromTableReorderingTest FromTableReorderingTest.cpp)
target_link_libraries(FromTableReorderingTest ${EXECUTE_TEST_LIBS} ${LibAwsS3_LIBRARIES})
add_test(FromTableReorderingTest FromTableReorderingTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS FromTableReorderingTest)

##########

add_executable(GeospatialTest GeospatialTest.cpp)
target_link_libraries(GeospatialTest ${EXECUTE_TEST_LIBS})
add_test(GeospatialTest GeospatialTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS GeospatialTest)

##########

if(NOT MSVC)
  add_executable(GeoTypesTest Geospatial/GeoTypesTest.cpp)
  target_link_libraries(GeoTypesTest gtest Logger Shared Geospatial ${CMAKE_DL_LIBS} ${LibAwsS3_LIBRARIES})
  add_test(GeoTypesTest GeoTypesTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS GeoTypesTest)
endif()

##########

if(ENABLE_CUDA)
  add_executable(GpuSharedMemoryTest GpuSharedMemoryTest.cpp ResultSetTestUtils.cpp)
  target_link_libraries(GpuSharedMemoryTest ${EXECUTE_TEST_LIBS})
  add_test(GpuSharedMemoryTest GpuSharedMemoryTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS GpuSharedMemoryTest)
endif()

##########

add_executable(GroupByTest GroupByTest.cpp)
target_link_libraries(GroupByTest ${EXECUTE_TEST_LIBS})
add_test(GroupByTest GroupByTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS GroupByTest)

##########

if(NOT MSVC)
  add_executable(ImportExportTest ImportExportTest.cpp)
  target_link_libraries(ImportExportTest gtest ${THRIFT_HANDLER_TEST_LIBRARIES})
  add_test(ImportExportTest ImportExportTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS ImportExportTest)
endif()

##########

add_executable(JoinHashTableTest JoinHashTableTest.cpp)
target_link_libraries(JoinHashTableTest ${EXECUTE_TEST_LIBS})
add_test(JoinHashTableTest JoinHashTableTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS JoinHashTableTest)

##########

if(NOT MSVC)
  add_executable(JSONTest JSONTest.cpp)
  target_link_libraries(JSONTest gtest Logger Shared)
  add_test(JSONTest JSONTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS JSONTest)
endif()

##########

add_executable(LoadTableTest LoadTableTest.cpp)
target_link_libraries(LoadTableTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(LoadTableTest LoadTableTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS LoadTableTest)

##########

add_executable(MigrationMgrTest MigrationMgrTest.cpp)
target_link_libraries(MigrationMgrTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(MigrationMgrTest MigrationMgrTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS MigrationMgrTest)

##########

if(ENABLE_SYSTEM_TFS)
  if (ENABLE_ML_ONEDAL_TFS OR ENABLE_ML_MLPACK_TFS)
    add_executable(MLFunctionsTest MLFunctionsTest.cpp)
    target_link_libraries(MLFunctionsTest ${EXECUTE_TEST_LIBS})
    add_test(MLFunctionsTest MLFunctionsTest ${TEST_ARGS})
    list(APPEND SANITY_TEST_PROGRAMS MLFunctionsTest)
  endif()
endif()

##########

if(NOT MSVC)
  add_executable(OmniSQLCommandTest OmniSQLCommandTest.cpp)
  target_link_libraries(OmniSQLCommandTest gtest ${Boost_LIBRARIES} mapd_thrift)
  if(ENABLE_KRB5)
    target_link_libraries(OmniSQLCommandTest krb5_gss Shared)
  endif()
  add_test(OmniSQLCommandTest OmniSQLCommandTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS OmniSQLCommandTest)
endif()

##########

if(NOT MSVC)
  add_executable(OmniSQLUtilitiesTest OmniSQLUtilitiesTest.cpp)
  target_link_libraries(OmniSQLUtilitiesTest gtest ${Boost_LIBRARIES})
  add_test(OmniSQLUtilitiesTest OmniSQLUtilitiesTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS OmniSQLUtilitiesTest)
endif()

##########

endif()  # Partition 3
if(${HEAVYAI_PARTITION} EQUAL 0 OR ${HEAVYAI_PARTITION} EQUAL 4)  # Partition 4

##########

add_executable(OverlapsJoinTest OverlapsJoinTest.cpp)
target_link_libraries(OverlapsJoinTest ${EXECUTE_TEST_LIBS})
add_test(OverlapsJoinTest OverlapsJoinTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS OverlapsJoinTest)

##########

add_executable(ParallelExecutorsTest ParallelExecutorsTest.cpp)
target_link_libraries(ParallelExecutorsTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ParallelExecutorsTest ParallelExecutorsTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ParallelExecutorsTest)
add_test(NAME ParallelExecutorsResourceMgrTest COMMAND ParallelExecutorsTest ${TEST_ARGS} "--enable-executor-resource-mgr")
list(APPEND SANITY_TESTS ParallelExecutorsResourceMgrTest)

##########

add_executable(PersistentStorageTest PersistentStorageTest.cpp)
target_link_libraries(PersistentStorageTest gtest ${MAPD_LIBRARIES} ${CURL_LIBRARY})
add_test(PersistentStorageTest PersistentStorageTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS PersistentStorageTest)

##########

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  add_executable(PkiEncryptorTest ee/PkiEncryptorTest.cpp)
  target_link_libraries(PkiEncryptorTest gtest Logger Shared)
  add_test(PkiEncryptorTest PkiEncryptorTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS PkiEncryptorTest)
endif()

##########

add_executable(ReadOnlyTest ReadOnlyTest.cpp)
target_link_libraries(ReadOnlyTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ReadOnlyTest ReadOnlyTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ReadOnlyTest)

##########

add_executable(ResultSetBaselineRadixSortTest ResultSetBaselineRadixSortTest.cpp ResultSetTestUtils.cpp)
target_link_libraries(ResultSetBaselineRadixSortTest ${EXECUTE_TEST_LIBS} ${LibAwsS3_LIBRARIES})
#target_link_libraries(ResultSetBaselineRadixSortTest gtest ${MAPD_LIBRARIES} ${AwsS3_CURL_SUPPORT})
add_test(ResultSetBaselineRadixSortTest ResultSetBaselineRadixSortTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ResultSetBaselineRadixSortTest)

##########

add_executable(ResultSetTest ResultSetTest.cpp ResultSetTestUtils.cpp)
target_link_libraries(ResultSetTest ${EXECUTE_TEST_LIBS})
add_test(ResultSetTest ResultSetTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ResultSetTest)

##########

if(ENABLE_SYSTEM_TFS)
  if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
    if(ENABLE_RF_PROP_TFS)
      add_executable(RFPropTableFunctionsTest ee/SystemTableFunctions/RFPropTableFunctionsTest.cpp)
      target_link_libraries(RFPropTableFunctionsTest ${EXECUTE_TEST_LIBS})
      add_test(RFPropTableFunctionsTest RFPropTableFunctionsTest ${TEST_ARGS})
      list(APPEND SANITY_TEST_PROGRAMS RFPropTableFunctionsTest)
    endif()
  endif()
endif()

##########

add_executable(RuntimeInterruptTest RuntimeInterruptTest.cpp)
target_link_libraries(RuntimeInterruptTest ${EXECUTE_TEST_LIBS})
add_test(RuntimeInterruptTest RuntimeInterruptTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS RuntimeInterruptTest)

##########

add_executable(ShardedTableEpochConsistencyTest ShardedTableEpochConsistencyTest.cpp)
target_link_libraries(ShardedTableEpochConsistencyTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ShardedTableEpochConsistencyTest ShardedTableEpochConsistencyTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ShardedTableEpochConsistencyTest)

##########

add_executable(ShowCommandsDdlTest ShowCommandsDdlTest.cpp)
target_link_libraries(ShowCommandsDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(ShowCommandsDdlTest ShowCommandsDdlTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ShowCommandsDdlTest)

##########

add_executable(SQLHintTest SQLHintTest.cpp)
target_link_libraries(SQLHintTest ${EXECUTE_TEST_LIBS})
add_test(SQLHintTest SQLHintTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS SQLHintTest)

##########

add_executable(SpecialCharsTest SpecialCharsTest.cpp)
target_link_libraries(SpecialCharsTest ${EXECUTE_TEST_LIBS})
add_test(SpecialCharsTest SpecialCharsTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS SpecialCharsTest)

##########

add_executable(StorageTest StorageTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
target_link_libraries(StorageTest gtest ${EXECUTE_TEST_LIBS})
add_test(StorageTest StorageTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS StorageTest)

##########

add_executable(StringDictionaryTest StringDictionaryTest.cpp)
if(ENABLE_FOLLY)
  target_link_libraries(StringDictionaryTest gtest mapd_thrift StringDictionary
    StringOps Logger Utils
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    ${Folly_LIBRARIES} ${Boost_LIBRARIES} ${ZLIB_LIBRARIES}
    ${AwsS3_CURL_SUPPORT} ${CMAKE_DL_LIBS})
else()
  target_link_libraries(StringDictionaryTest gtest mapd_thrift StringDictionary
    StringOps Logger Utils
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${AwsS3_CURL_SUPPORT} ${CMAKE_DL_LIBS})
endif()
add_test(StringDictionaryTest StringDictionaryTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS StringDictionaryTest)
add_test(NAME StringDictionaryHashTest COMMAND StringDictionaryTest ${TEST_ARGS} "--enable-string-dict-hash-cache")
list(APPEND SANITY_TESTS StringDictionaryHashTest)

##########

add_executable(StringFunctionsTest StringFunctionsTest.cpp)
target_link_libraries(StringFunctionsTest ${EXECUTE_TEST_LIBS})
add_test(StringFunctionsTest StringFunctionsTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS StringFunctionsTest)

##########

if(ENABLE_SYSTEM_TFS)
  add_executable(SystemTableFunctionsTest SystemTableFunctionsTest.cpp)
  target_link_libraries(SystemTableFunctionsTest ${EXECUTE_TEST_LIBS})
  add_test(SystemTableFunctionsTest SystemTableFunctionsTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS SystemTableFunctionsTest)
endif()

##########

add_executable(TableFunctionsTest TableFunctionsTest.cpp)
target_link_libraries(TableFunctionsTest ${EXECUTE_TEST_LIBS})
add_test(TableFunctionsTest TableFunctionsTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS TableFunctionsTest)

##########

add_executable(ThreadingTest Shared/ThreadingTest.cpp)
target_link_libraries(ThreadingTest ${CMAKE_DL_LIBS} gtest Logger Shared ${LLVM_LINKER_FLAGS} ${TBB_LIBRARIES})
add_test(ThreadingTest ThreadingTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ThreadingTest)

##########

add_executable(ThreadingTestSTD Shared/ThreadingTest.cpp)
target_compile_options(ThreadingTestSTD PRIVATE -UENABLE_TBB -DENABLE_TBB=0)
target_link_libraries(ThreadingTestSTD gtest Logger Shared ${LLVM_LINKER_FLAGS} ${CMAKE_DL_LIBS})
add_test(ThreadingTestSTD ThreadingTestSTD ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS ThreadingTestSTD)

##########

add_executable(TokenCompletionHintsTest TokenCompletionHintsTest.cpp)
target_link_libraries(TokenCompletionHintsTest token_completion_hints gtest mapd_thrift Logger Shared ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
add_test(TokenCompletionHintsTest TokenCompletionHintsTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS TokenCompletionHintsTest)

##########

add_executable(TopKTest TopKTest.cpp)
target_link_libraries(TopKTest ${EXECUTE_TEST_LIBS})
add_test(TopKTest TopKTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS TopKTest)

##########

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  add_executable(UdfTest UdfTest.cpp)
  target_link_libraries(UdfTest gtest UdfCompiler ${EXECUTE_TEST_LIBS})
  add_test(UdfTest UdfTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS UdfTest)
endif()

##########

add_executable(UpdateMetadataTest UpdateMetadataTest.cpp)
target_link_libraries(UpdateMetadataTest ${EXECUTE_TEST_LIBS})
add_test(UpdateMetadataTest UpdateMetadataTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS UpdateMetadataTest)

##########

add_executable(UpdelStorageTest UpdelStorageTest.cpp)
target_link_libraries(UpdelStorageTest gtest ${EXECUTE_TEST_LIBS})
add_test(UpdelStorageTest UpdelStorageTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS UpdelStorageTest)

##########

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  add_executable(UserMappingDdlTest ee/UserMappingDdlTest.cpp)
  target_link_libraries(UserMappingDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
  add_test(UserMappingDdlTest UserMappingDdlTest ${TEST_ARGS})
  list(APPEND SANITY_TEST_PROGRAMS UserMappingDdlTest)
endif()

##########

add_executable(SessionsStoreTest SessionsStoreTest.cpp)
target_link_libraries(SessionsStoreTest ${THRIFT_HANDLER_TEST_LIBRARIES})
add_test(SessionsStoreTest SessionsStoreTest ${TEST_ARGS})
list(APPEND SANITY_TEST_PROGRAMS SessionsStoreTest)

##########

endif()  # Partition 4

##########

########## Other tests (non-sanity_tests).

##########

if(NOT MSVC)
  add_executable(ArrowIpcIntegrationTest ArrowIpcIntegrationTest.cpp)
  target_link_libraries(ArrowIpcIntegrationTest gtest fmt::fmt ThriftClient mapd_thrift Logger Shared ${CUDA_LIBRARIES} ${Arrow_LIBRARIES})
  add_test(ArrowIpcIntegrationTest ArrowIpcIntegrationTest ${TEST_ARGS})
endif()

##########

if(NOT MSVC)
  add_executable(CtasIntegrationTest CtasIntegrationTest.cpp)
  target_link_libraries(CtasIntegrationTest gtest Logger Shared mapd_thrift ThriftClient ${LLVM_LINKER_FLAGS})
endif()

##########


if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  add_executable(MultiInstanceTest ee/MultiInstanceTest.cpp)
  target_link_libraries(MultiInstanceTest gtest ${MAPD_LIBRARIES} ${CURL_LIBRARIES})
  add_dependencies(MultiInstanceTest thrift_gen)
  add_dependencies(MultiInstanceTest heavydb) # test invokes heavydb bin directly
  add_dependencies(MultiInstanceTest mapd_java_components)
  add_test(MultiInstanceTest, MultiInstanceTest ${TEST_ARGS})
endif()

##########

add_executable(GeospatialBenchmark GeospatialBenchmark.cpp)
target_link_libraries(GeospatialBenchmark benchmark ${EXECUTE_TEST_LIBS})

##########

if(NOT MSVC)
  add_executable(QuantileCpuTest Quantile/QuantileCpuTest.cpp)
  target_link_libraries(QuantileCpuTest gtest ${MAPD_LIBRARIES})
endif()

##########

add_executable(RunQueryLoop RunQueryLoop.cpp)
target_link_libraries(RunQueryLoop ${EXECUTE_TEST_LIBS} rg3_bcrypt)
add_test(RunQueryLoop RunQueryLoop ${TEST_ARGS})

##########

if("${MAPD_EDITION_LOWER}" STREQUAL "ee" AND ENABLE_AWS_S3)
  add_executable(S3SelectClientTest ee/S3SelectClientTest.cpp)
  target_link_libraries(S3SelectClientTest ${EXECUTE_TEST_LIBS})
  add_test(S3SelectClientTest S3SelectClientTest ${TEST_ARGS})
endif()

##########

add_executable(StoragePerfTest StoragePerfTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
target_link_libraries(StoragePerfTest gtest ${EXECUTE_TEST_LIBS})
add_test(StoragePerfTest StoragePerfTest ${TEST_ARGS})

##########

add_executable(StringDictionaryBenchmark StringDictionaryBenchmark.cpp)
if(ENABLE_FOLLY)
  target_link_libraries(StringDictionaryBenchmark benchmark gtest mapd_thrift StringDictionary StringOps Logger Utils $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs> ${Folly_LIBRARIES} ${Boost_LIBRARIES} ${ZLIB_LIBRARIES})
else()
  target_link_libraries(StringDictionaryBenchmark benchmark gtest mapd_thrift StringDictionary StringOps Logger Utils $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs> ${Boost_LIBRARIES} ${ZLIB_LIBRARIES})
endif()

##########

add_executable(StringTransformTest StringTransformTest.cpp)
target_link_libraries(StringTransformTest Logger Shared gtest ${Boost_LIBRARIES})
add_test(StringTransformTest StringTransformTest ${TEST_ARGS})

##########

add_executable(TableUpdateDeleteBenchmark TableUpdateDeleteBenchmark.cpp)
target_link_libraries(TableUpdateDeleteBenchmark benchmark ${EXECUTE_TEST_LIBS})

##########

add_executable(UtilTest UtilTest.cpp)
target_link_libraries(UtilTest Utils gtest Logger Shared ${Boost_LIBRARIES} OSDependent)
add_test(UtilTest UtilTest ${TEST_ARGS})

##########

########## Build targets.

##########

add_custom_target(clean-unit-tests-tmp-dir
    COMMAND rm -rf ${TEST_BASE_PATH}
    USES_TERMINAL)
add_dependencies(clean-all clean-unit-tests-tmp-dir)

add_custom_target(init_test_dir
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initheavy -f ${TEST_BASE_PATH} || exit 0
    USES_TERMINAL)

if (ENABLE_INIT_DIR)
  foreach(PROGRAM ${SANITY_TEST_PROGRAMS} ${INSANITY_TEST_PROGRAMS})
    add_dependencies(${PROGRAM} init_test_dir)
  endforeach()
endif()

list(APPEND SANITY_TESTS ${SANITY_TEST_PROGRAMS})
set_tests_properties(${SANITY_TESTS} PROPERTIES LABELS "_sanity_")

add_custom_target(sanity_tests_build_only DEPENDS init_test_dir ${SANITY_TEST_PROGRAMS})

add_custom_target(sanity_tests
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose --label-regex "_sanity_"
    DEPENDS init_test_dir ${SANITY_TEST_PROGRAMS}
    USES_TERMINAL)

if(${HEAVYAI_PARTITION} EQUAL 0 OR ${HEAVYAI_PARTITION} EQUAL 1)  # Partition 1

# list(APPEND INSANITY_TESTS ${INSANITY_TEST_PROGRAMS})
set_tests_properties(${INSANITY_TESTS} PROPERTIES LABELS "_insanity_")

add_custom_target(insanity_tests
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose --label-regex "\"_sanity_|_insanity_\""
    DEPENDS init_test_dir ${SANITY_TEST_PROGRAMS} ${INSANITY_TEST_PROGRAMS}
    USES_TERMINAL)

endif()  # Partition 1

add_custom_target(all_tests
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS init_test_dir ${SANITY_TEST_PROGRAMS} ${INSANITY_TEST_PROGRAMS} UtilTest RunQueryLoop StringDictionaryTest StringTransformTest StoragePerfTest
    USES_TERMINAL)

add_custom_target(storage_perf_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --tests-regex "\"(StoragePerfTest)\""
    DEPENDS init_test_dir StoragePerfTest)

add_custom_target(topk_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --tests-regex "\"(TopKTest)\""
    DEPENDS init_test_dir TopKTest)

if(ENABLE_DBE)
  set(DBE_TEST_PROGRAMS EmbeddedDatabaseTest)
  set_tests_properties(${DBE_TEST_PROGRAMS} PROPERTIES LABELS "embedded")
  add_custom_target(dbe_all_tests
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose --label-regex embedded
    DEPENDS init_test_dir DBEngine ${DBE_TEST_PROGRAMS}
    USES_TERMINAL)
  add_dependencies(dbe_all_tests dbe4py-test)
  add_dependencies(all_tests dbe4py-test)
endif()

add_subdirectory(ExtensionFunctions/h3lib/)
