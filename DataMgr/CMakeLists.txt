set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")

set(datamgr_source_files
    AbstractBuffer.cpp
    Allocators/CudaAllocator.cpp
    Allocators/ThrustAllocator.cpp
    Chunk/Chunk.cpp
    DataMgr.cpp
    Encoder.cpp
    StringNoneEncoder.cpp
    FileMgr/GlobalFileMgr.cpp
    FileMgr/FileMgr.cpp
    FileMgr/FileBuffer.cpp
    FileMgr/FileInfo.cpp
    ForeignStorage/ArrowCsvForeignStorage.cpp
    ForeignStorage/CsvDataWrapper.cpp
    ForeignStorage/DummyForeignStorage.cpp
    ForeignStorage/ForeignStorageInterface.cpp
    ForeignStorage/ForeignStorageBuffer.cpp
    ForeignStorage/ForeignStorageMgr.cpp
    ForeignStorage/ForeignStorageCache.cpp
    ForeignStorage/CacheEvictionAlgorithms/LRUEvictionAlgorithm.cpp
    ForeignStorage/CsvReader.cpp
    BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.cpp
    BufferMgr/GpuCudaBufferMgr/GpuCudaBuffer.cpp
    BufferMgr/CpuBufferMgr/CpuBufferMgr.cpp
    BufferMgr/CpuBufferMgr/CpuBuffer.cpp
    BufferMgr/BufferMgr.cpp
    BufferMgr/Buffer.cpp
    ForeignStorage/ForeignStorageBuffer.cpp
    ForeignStorage/ForeignStorageMgr.cpp
    ForeignStorage/CsvDataWrapper.cpp
    ForeignStorage/ParquetDataWrapper.cpp
    PersistentStorageMgr/PersistentStorageMgr.cpp
    ForeignStorage/LazyParquetImporter.cpp
    ForeignStorage/LazyParquetChunkLoader.cpp
    ForeignStorage/ParquetShared.cpp
)

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
if(ENABLE_AWS_S3)
  list(APPEND datamgr_source_files ForeignStorage/ee/CsvReaderS3.cpp)
endif()
endif()

add_library(DataMgr ${datamgr_source_files})

target_link_libraries(DataMgr CudaMgr Shared ${Boost_THREAD_LIBRARY} ${TBB_LIBS})

option(ENABLE_CRASH_CORRUPTION_TEST "Enable crash using SIGUSR2 during page deletion to faster and affirmative test/repro db corruption" OFF)
if(ENABLE_CRASH_CORRUPTION_TEST)
  add_definitions("-DENABLE_CRASH_CORRUPTION_TEST")
endif()
