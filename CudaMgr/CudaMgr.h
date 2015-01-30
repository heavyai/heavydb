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
    


struct CudaMgr {

    CudaMgr();
    ~CudaMgr();
    void fillDeviceProperties();
    void createDeviceContexts();
    void printDeviceProperties();
    void checkError(CUresult cuResult);
    int deviceCount;
    std::vector <DeviceProperties> deviceProperties;
    std::vector <CUcontext> deviceContexts;


}; //class CudaMgr

} // Namespace CudaMgr_Namespace

#endif // CUDAMGR_H
