#ifndef CUDAMGR_H
#define CUDAMGR_H

#include <vector>
#include <cuda.h>

namespace CudaMgr_Namespace {

struct DeviceProperties {
    CUdevice device;
    int computeMajor;
    int computeMinor;
    size_t globalMem;
    int constantMem;
    int sharedMemPerBlock;
    int numMPs;
    int warpSize;
    int maxRegistersPerBlock;
    int maxRegistersPerMP;
    int pciBusId;
    int pciDeviceId;
    int memoryClockKhz;
    int memoryBusWidth; // in bits
    float memoryBandwidthGBs;
};
    


class CudaMgr {

    public:
        CudaMgr();
        ~CudaMgr();
    void setContext(const int deviceNum);
    void printDeviceProperties();
    int8_t * allocatePinnedHostMem(const size_t numBytes);
    int8_t * allocateDeviceMem(const size_t numBytes, const int deviceNum);
    void freePinnedHostMem(int8_t * hostPtr);
    void freeDeviceMem(int8_t * devicePtr);
    void copyHostToDevice(int8_t *devicePtr, const int8_t *hostPtr, const size_t numBytes, const int deviceNum);
    void copyDeviceToHost(int8_t *hostPtr, const int8_t *devicePtr, const size_t numBytes, const int deviceNum);

    inline int getDeviceCount() {return deviceCount;}

    private: 
        void fillDeviceProperties();
        void createDeviceContexts();
        void checkError(CUresult cuResult);

        int deviceCount;
        std::vector <DeviceProperties> deviceProperties;
        std::vector <CUcontext> deviceContexts;


}; //class CudaMgr

} // Namespace CudaMgr_Namespace

#endif // CUDAMGR_H
