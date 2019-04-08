find_package(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")

set(datamgr_source_files
    DataMgr.cpp
    Encoder.cpp
    StringNoneEncoder.cpp
    FileMgr/GlobalFileMgr.cpp
    FileMgr/FileMgr.cpp
    FileMgr/FileBuffer.cpp
    FileMgr/FileInfo.cpp
    BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.cpp
    BufferMgr/GpuCudaBufferMgr/GpuCudaBuffer.cpp
    BufferMgr/CpuBufferMgr/CpuBufferMgr.cpp
    BufferMgr/CpuBufferMgr/CpuBuffer.cpp
    BufferMgr/BufferMgr.cpp
    BufferMgr/Buffer.cpp
    LockMgr.cpp
)

add_library(DataMgr ${datamgr_source_files})

target_link_libraries(DataMgr CudaMgr Shared ${Boost_THREAD_LIBRARY} ${Glog_LIBRARIES})

option(ENABLE_CRASH_CORRUPTION_TEST "Enable crash using SIGUSR2 during page deletion to faster and affirmative test/repro db corruption" OFF)
if(ENABLE_CRASH_CORRUPTION_TEST)
  add_definitions("-DENABLE_CRASH_CORRUPTION_TEST")
endif()
