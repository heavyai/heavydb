if(NOT MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
endif()

set(datamgr_source_files
    AbstractBuffer.cpp
    Allocators/CudaAllocator.cpp
    Allocators/ThrustAllocator.cpp
    Chunk/Chunk.cpp
    DataMgr.cpp
    DataMgrBufferProvider.cpp
    DataMgrDataProvider.cpp
    Encoder.cpp
    StringNoneEncoder.cpp
    BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.cpp
    BufferMgr/GpuCudaBufferMgr/GpuCudaBuffer.cpp
    BufferMgr/CpuBufferMgr/CpuBufferMgr.cpp
    BufferMgr/CpuBufferMgr/CpuBuffer.cpp
    BufferMgr/CpuBufferMgr/TieredCpuBufferMgr.cpp
    BufferMgr/BufferMgr.cpp
    BufferMgr/Buffer.cpp
    PersistentStorageMgr/PersistentStorageMgr.cpp
)

# On Linux, Arrow internally links symbols for its dependency libraries (such as the C++ AWS SDK)
# when it is built as a shared library. In this case, set a flag that would be used to determine
# how to handle AWS SDK initialization and shutdown.
function(set_arrow_has_private_aws_sdk)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    foreach(LIB_PATH ${Arrow_LIBRARIES})
      get_filename_component(LIB_NAME ${LIB_PATH} NAME)
      if (${LIB_NAME} STREQUAL "${CMAKE_SHARED_LIBRARY_PREFIX}arrow${CMAKE_SHARED_LIBRARY_SUFFIX}")
        add_definitions("-DARROW_HAS_PRIVATE_AWS_SDK")
      endif()
    endforeach()
  endif()
endfunction()

if(ENABLE_AWS_S3)
  list(APPEND datamgr_source_files OmniSciAwsSdk.cpp)
  set_arrow_has_private_aws_sdk()
endif()

add_library(DataMgr ${datamgr_source_files})

target_link_libraries(DataMgr CudaMgr Shared SchemaMgr ${Boost_THREAD_LIBRARY} ${TBB_LIBS} ${CMAKE_DL_LIBS})

option(ENABLE_CRASH_CORRUPTION_TEST "Enable crash using SIGUSR2 during page deletion to faster and affirmative test/repro db corruption" OFF)
if(ENABLE_CRASH_CORRUPTION_TEST)
  add_definitions("-DENABLE_CRASH_CORRUPTION_TEST")
endif()
