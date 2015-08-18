#include "CudaUtils.h"
#include <cuda.h>

namespace CudaUtils {

template <typename T>
void allocGpuMem(T*& devMem, const size_t numElems, const size_t elemSize, const int gpuNum) {
  cudaSetDevice(gpuNum);
  cudaMalloc((void**)&devMem, numElems * elemSize);
}

template <typename T>
void allocPinnedHostMem(T*& hostMem, const size_t numElems, const size_t elemSize) {
  cudaHostAlloc((void**)&hostMem, numElems * elemSize, cudaHostAllocPortable);
}

template <typename T>
void copyToGpu(T* devMem, const T* hostMem, const size_t numElems, const size_t elemSize, const int gpuNum) {
  cudaSetDevice(gpuNum);
  cudaMemcpy(devMem, hostMem, numElems * elemSize, cudaMemcpyHostToDevice);
}

template <typename T>
void copyGpuToGpu(T* dstMem,
                  const T* srcMem,
                  const std::size_t numElems,
                  const std::size_t elemSize,
                  const int dstGpuNum) {
  cudaSetDevice(dstGpuNum);
  cudaMemcpy(dstMem, srcMem, numElems * elemSize, cudaMemcpyDefault);
}

template <typename T>
void copyToHost(T* hostMem, const T* devMem, const size_t numElems, const size_t elemSize, const int gpuNum) {
  cudaSetDevice(gpuNum);
  cudaMemcpy(hostMem, devMem, numElems * elemSize, cudaMemcpyDeviceToHost);
}

template <typename T>
void gpuFree(T*& devMem) {
  cudaFree(devMem);
  devMem = 0;
}

template <typename T>
void hostFree(T*& hostMem) {
  cudaFreeHost(hostMem);
  hostMem = 0;
}

template void allocGpuMem<bool>(bool*& devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
template void allocGpuMem<char>(char*& devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
template void allocGpuMem<unsigned char>(unsigned char*& devMem,
                                         const size_t numElems,
                                         const size_t elemSize,
                                         const int gpuNum);
template void allocGpuMem<int>(int*& devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
template void allocGpuMem<int8_t>(int8_t*& devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
template void allocGpuMem<unsigned int>(unsigned int*& devMem,
                                        const size_t numElems,
                                        const size_t elemSize,
                                        const int gpuNum);
template void allocGpuMem<unsigned long>(unsigned long*& devMem,
                                         const size_t numElems,
                                         const size_t elemSize,
                                         const int gpuNum);
template void allocGpuMem<unsigned long long int>(unsigned long long int*& devMem,
                                                  const size_t numElems,
                                                  const size_t elemSize,
                                                  const int gpuNum);
template void allocGpuMem<float>(float*& devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
template void allocGpuMem<double>(double*& devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
// template void allocGpuMem <void>(void * &devMem, const size_t numElems, const size_t elemSize);

template void allocPinnedHostMem<int>(int*& hostMem, const size_t numElems, const size_t elemSize);
template void allocPinnedHostMem<int8_t>(int8_t*& hostMem, const size_t numElems, const size_t elemSize);
template void allocPinnedHostMem<char>(char*& hostMem, const size_t numElems, const size_t elemSize);
template void allocPinnedHostMem<unsigned char>(unsigned char*& hostMem, const size_t numElems, const size_t elemSize);
template void allocPinnedHostMem<float>(float*& hostMem, const size_t numElems, const size_t elemSize);
template void allocPinnedHostMem<unsigned int>(unsigned int*& hostMem, const size_t numElems, const size_t elemSize);
template void allocPinnedHostMem<void>(void*& hostMem, const size_t numElems, const size_t elemSize);

template void copyToGpu<bool>(bool* devMem,
                              const bool* hostMem,
                              const size_t numElems,
                              const size_t elemSize,
                              const int gpuNum);
template void copyToGpu<char>(char* devMem,
                              const char* hostMem,
                              const size_t numElems,
                              const size_t elemSize,
                              const int gpuNum);
template void copyToGpu<unsigned char>(unsigned char* devMem,
                                       const unsigned char* hostMem,
                                       const size_t numElems,
                                       const size_t elemSize,
                                       const int gpuNum);
template void copyToGpu<int>(int* devMem,
                             const int* hostMem,
                             const size_t numElems,
                             const size_t elemSize,
                             const int gpuNum);
template void copyToGpu<int8_t>(int8_t* devMem,
                                const int8_t* hostMem,
                                const size_t numElems,
                                const size_t elemSize,
                                const int gpuNum);
template void copyToGpu<unsigned int>(unsigned int* devMem,
                                      const unsigned int* hostMem,
                                      const size_t numElems,
                                      const size_t elemSize,
                                      const int gpuNum);
template void copyToGpu<unsigned long>(unsigned long* devMem,
                                       const unsigned long* hostMem,
                                       const size_t numElems,
                                       const size_t elemSize,
                                       const int gpuNum);
template void copyToGpu<unsigned long long int>(unsigned long long int* devMem,
                                                const unsigned long long int* hostMem,
                                                const size_t numElems,
                                                const size_t elemSize,
                                                const int gpuNum);
template void copyToGpu<float>(float* devMem,
                               const float* hostMem,
                               const size_t numElems,
                               const size_t elemSize,
                               const int gpuNum);
template void copyToGpu<double>(double* devMem,
                                const double* hostMem,
                                const size_t numElems,
                                const size_t elemSize,
                                const int gpuNum);

template void copyGpuToGpu<bool>(bool* dstMem,
                                 const bool* srcMem,
                                 const std::size_t numElems,
                                 const std::size_t elemSize,
                                 const int dstGpuNum);
template void copyGpuToGpu<char>(char* dstMem,
                                 const char* srcMem,
                                 const std::size_t numElems,
                                 const std::size_t elemSize,
                                 const int dstGpuNum);
template void copyGpuToGpu<unsigned char>(unsigned char* dstMem,
                                          const unsigned char* srcMem,
                                          const std::size_t numElems,
                                          const std::size_t elemSize,
                                          const int dstGpuNum);
template void copyGpuToGpu<int>(int* dstMem,
                                const int* srcMem,
                                const std::size_t numElems,
                                const std::size_t elemSize,
                                const int dstGpuNum);
template void copyGpuToGpu<int8_t>(int8_t* dstMem,
                                   const int8_t* srcMem,
                                   const std::size_t numElems,
                                   const std::size_t elemSize,
                                   const int dstGpuNum);
template void copyGpuToGpu<unsigned int>(unsigned int* dstMem,
                                         const unsigned int* srcMem,
                                         const std::size_t numElems,
                                         const std::size_t elemSize,
                                         const int dstGpuNum);
template void copyGpuToGpu<unsigned long long int>(unsigned long long int* dstMem,
                                                   const unsigned long long int* srcMem,
                                                   const std::size_t numElems,
                                                   const std::size_t elemSize,
                                                   const int dstGpuNum);
template void copyGpuToGpu<float>(float* dstMem,
                                  const float* srcMem,
                                  const std::size_t numElems,
                                  const std::size_t elemSize,
                                  const int dstGpuNum);
template void copyGpuToGpu<double>(double* dstMem,
                                   const double* srcMem,
                                   const std::size_t numElems,
                                   const std::size_t elemSize,
                                   const int dstGpuNum);

// template void copyToHost <__nv_bool> (__nv_bool * hostMem, __nv_bool * devMem, const size_t numElems, const size_t
// elemSize, const int gpuNum);

template void copyToHost<bool>(bool* hostMem,
                               const bool* devMem,
                               const size_t numElems,
                               const size_t elemSize,
                               const int gpuNum);
template void copyToHost<char>(char* hostMem,
                               const char* devMem,
                               const size_t numElems,
                               const size_t elemSize,
                               const int gpuNum);
template void copyToHost<unsigned char>(unsigned char* hostMem,
                                        const unsigned char* devMem,
                                        const size_t numElems,
                                        const size_t elemSize,
                                        const int gpuNum);
template void copyToHost<unsigned short>(unsigned short* hostMem,
                                         const unsigned short* devMem,
                                         const size_t numElems,
                                         const size_t elemSize,
                                         const int gpuNum);
template void copyToHost<int>(int* hostMem,
                              const int* devMem,
                              const size_t numElems,
                              const size_t elemSize,
                              const int gpuNum);
template void copyToHost<int8_t>(int8_t* hostMem,
                                 const int8_t* devMem,
                                 const size_t numElems,
                                 const size_t elemSize,
                                 const int gpuNum);
template void copyToHost<unsigned int>(unsigned int* hostMem,
                                       const unsigned int* devMem,
                                       const size_t numElems,
                                       const size_t elemSize,
                                       const int gpuNum);
template void copyToHost<unsigned long long int>(unsigned long long int* hostMem,
                                                 const unsigned long long int* devMem,
                                                 const size_t numElems,
                                                 const size_t elemSize,
                                                 const int gpuNum);
template void copyToHost<float>(float* hostMem,
                                const float* devMem,
                                const size_t numElems,
                                const size_t elemSize,
                                const int gpuNum);
template void copyToHost<double>(double* hostMem,
                                 const double* devMem,
                                 const size_t numElems,
                                 const size_t elemSize,
                                 const int gpuNum);
template void copyToHost<void>(void* hostMem,
                               const void* devMem,
                               const size_t numElems,
                               const size_t elemSize,
                               const int gpuNum);

template void gpuFree<bool>(bool*& devMem);
template void gpuFree<char>(char*& devMem);
template void gpuFree<int>(int*& devMem);
template void gpuFree<int8_t>(int8_t*& devMem);
template void gpuFree<unsigned int>(unsigned int*& devMem);
template void gpuFree<unsigned long>(unsigned long*& devMem);
template void gpuFree<unsigned long long int>(unsigned long long int*& devMem);
template void gpuFree<float>(float*& devMem);
template void gpuFree<double>(double*& devMem);
template void gpuFree<unsigned char>(unsigned char*& devMem);
template void gpuFree<void>(void*& devMem);

template void hostFree<bool>(bool*& hostMem);
template void hostFree<char>(char*& hostMem);
template void hostFree<int>(int*& hostMem);
template void hostFree<int8_t>(int8_t*& hostMem);
template void hostFree<unsigned int>(unsigned int*& hostMem);
template void hostFree<unsigned long long int>(unsigned long long int*& hostMem);
template void hostFree<float>(float*& hostMem);
template void hostFree<double>(double*& hostMem);
template void hostFree<unsigned char>(unsigned char*& hostMem);
// template void hostFree <geops_size_t> (geops_size_t * &hostMem);
template void hostFree<void>(void*& hostMem);
}
