cmake_minimum_required(VERSION 3.16)
project(NvidiaComputeCapability)

include_directories(..)

enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY})

add_executable(get_nvidia_compute_capability get_nvidia_compute_capability.cpp)
target_link_libraries(get_nvidia_compute_capability ${CUDA_LIBRARIES})
add_custom_command(
  COMMENT
    "Detecting NVIDIA CUDA compute capability for debug build. (NvidiaComputeCapability.txt)"
  OUTPUT
    ${CMAKE_BINARY_DIR}/../NvidiaComputeCapability.txt
  DEPENDS
    get_nvidia_compute_capability
  COMMAND
    get_nvidia_compute_capability > ${CMAKE_BINARY_DIR}/../NvidiaComputeCapability.txt
  COMMAND
    eval echo "NVIDIA CUDA compute capability: $(cat ${CMAKE_BINARY_DIR}/../NvidiaComputeCapability.txt)"
  VERBATIM
)
add_custom_target(NvidiaComputeCapability ALL DEPENDS ${CMAKE_BINARY_DIR}/../NvidiaComputeCapability.txt)
