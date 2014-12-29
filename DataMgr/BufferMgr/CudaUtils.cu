#include "CudaUtils.h"
#include <cuda.h>

namespace CudaUtils {
 
    template <typename T> void allocGpuMem(T * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum) {
        cudaSetDevice(gpuNum);
        cudaMalloc((void**)&devMem, numElems * elemSize);
    }

    template <typename T> void allocPinnedHostMem(T * &hostMem, const size_t numElems, const size_t elemSize) {
    cudaHostAlloc((void**)&hostMem, numElems * elemSize, cudaHostAllocPortable);
}

    template <typename T> void copyToGpu(const T* hostMem, T* devMem, const size_t numElems, const size_t elemSize, const int gpuNum) {
        cudaSetDevice(gpuNum);
        cudaMemcpy(devMem,hostMem,numElems * elemSize, cudaMemcpyHostToDevice); 
    }

    template <typename T> void copyToHost (T* hostMem, T* devMem, const size_t numElems, const size_t elemSize, const int gpuNum) {
        cudaSetDevice(gpuNum);
        cudaMemcpy(hostMem,devMem,numElems * elemSize, cudaMemcpyDeviceToHost); 
    }

    template <typename T> void gpuFree (T *&devMem) {
        cudaFree(devMem);
        devMem = 0;
    }

    template <typename T> void hostFree (T *&hostMem) {
        cudaFreeHost(hostMem);
        hostMem = 0;
    }


    template void allocGpuMem <bool>(bool * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void allocGpuMem <char>(char * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void allocGpuMem <unsigned char>(unsigned char * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void allocGpuMem <int>(int * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void allocGpuMem <unsigned int>(unsigned int * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void allocGpuMem <unsigned long>(unsigned long* &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void allocGpuMem <unsigned long long int>(unsigned long long int * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void allocGpuMem <float>(float * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void allocGpuMem <double>(double * &devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    //template void allocGpuMem <void>(void * &devMem, const size_t numElems, const size_t elemSize);

    template void allocPinnedHostMem <int>(int * &hostMem, const size_t numElems, const size_t elemSize);
    template void allocPinnedHostMem <char>(char * &hostMem, const size_t numElems, const size_t elemSize);
    template void allocPinnedHostMem <unsigned char>(unsigned char * &hostMem, const size_t numElems, const size_t elemSize);
    template void allocPinnedHostMem <float>(float * &hostMem, const size_t numElems, const size_t elemSize);
    template void allocPinnedHostMem <unsigned int>(unsigned int * &hostMem, const size_t numElems, const size_t elemSize);
    template void allocPinnedHostMem <void>(void * &hostMem, const size_t numElems, const size_t elemSize);

    template void copyToGpu <bool> (const bool* hostMem, bool* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToGpu <char> (const char* hostMem, char* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToGpu <unsigned char> (const unsigned char* hostMem, unsigned char* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToGpu <int> (const int* hostMem, int* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToGpu <unsigned int> (const unsigned int* hostMem, unsigned int* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToGpu <unsigned long> (const unsigned long* hostMem, unsigned long* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToGpu <unsigned long long int> (const unsigned long long int* hostMem, unsigned long long int* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToGpu <float> (const float* hostMem, float* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToGpu <double> (const double* hostMem, double* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);

    //template void copyToHost <__nv_bool> (__nv_bool * hostMem, __nv_bool * devMem, const size_t numElems, const size_t elemSize, const int gpuNum);

    template void copyToHost <bool> (bool* hostMem, bool* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <char> (char* hostMem, char* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <unsigned char> (unsigned char* hostMem, unsigned char* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <unsigned short> (unsigned short* hostMem, unsigned short* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <int> (int* hostMem, int* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <unsigned int> (unsigned int* hostMem, unsigned int* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <unsigned long long int> (unsigned long long int* hostMem, unsigned long long int* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <float> (float* hostMem, float* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <double> (double* hostMem, double* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);
    template void copyToHost <void> (void* hostMem, void* devMem, const size_t numElems, const size_t elemSize, const int gpuNum);

    template void gpuFree <bool> (bool * &devMem);
    template void gpuFree <char> (char * &devMem);
    template void gpuFree <int> (int * &devMem);
    template void gpuFree <unsigned int> (unsigned int * &devMem);
    template void gpuFree <unsigned long> (unsigned long * &devMem);
    template void gpuFree <unsigned long long int> (unsigned long long int * &devMem);
    template void gpuFree <float> (float * &devMem);
    template void gpuFree <double> (double * &devMem);
    template void gpuFree <unsigned char> (unsigned char * &devMem);
    template void gpuFree <void> (void * &devMem);

    template void hostFree <bool> (bool * &hostMem);
    template void hostFree <char> (char * &hostMem);
    template void hostFree <int> (int * &hostMem);
    template void hostFree <unsigned int> (unsigned int * &hostMem);
    template void hostFree <unsigned long long int> (unsigned long long int * &hostMem);
    template void hostFree <float> (float * &hostMem);
    template void hostFree <double> (double * &hostMem);
    template void hostFree <unsigned char> (unsigned char * &hostMem);
    //template void hostFree <geops_size_t> (geops_size_t * &hostMem);
    template void hostFree <void> (void * &hostMem);
}

