#ifndef CUDAUTILS_H
#define CUDAUTILS_H


namespace CudaUtils {
    
    template <typename T> void allocGpuMem(T * &devMem, const std::size_t numElems, const std::size_t elemSize, const int gpuNum);

    template <typename T> void allocPinnedHostMem(T * &hostMem, const std::size_t numElems, const std::size_t elemSize);

    template <typename T> void copyToGpu(T* devMem, const T* hostMem, const std::size_t numElems, const std::size_t elemSize, const int gpuNum);

    template <typename T>  void copyToHost (T* hostMem, const T* devMem, const std::size_t numElems, const std::size_t elemSize, const int gpuNum);

    template <typename T>  void copyGpuToGpu (T* dstMem, const T* srcMem, const std::size_t numElems, const std::size_t elemSize, const int dstGpuNum);

    template <typename T>  void gpuFree (T* &devMem);
    
    template <typename T> void hostFree (T * &hostMem);

}

#endif
