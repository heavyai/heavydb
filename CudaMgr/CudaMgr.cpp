#include "CudaMgr.h"
#include <stdexcept>
#include <iostream>
#include "assert.h"

using std::cout;
using std::endl;

namespace CudaMgr_Namespace {

CudaMgr::CudaMgr() {
    checkError(cuInit(0));
    checkError(cuDeviceGetCount(&deviceCount));
    fillDeviceProperties();
    createDeviceContexts();
}

CudaMgr::~CudaMgr() {
    for (int d = 0; d < deviceCount; ++d) {
        checkError(cuCtxDestroy(deviceContexts[d]));
    }
}

void CudaMgr::fillDeviceProperties() {
    deviceProperties.resize(deviceCount);
    for (int deviceNum = 0; deviceNum < deviceCount; ++deviceNum) {
        checkError(cuDeviceGet(&deviceProperties[deviceNum].device,deviceNum));
        checkError(cuDeviceComputeCapability(&deviceProperties[deviceNum].computeMajor, &deviceProperties[deviceNum].computeMinor, deviceProperties[deviceNum].device));
        checkError(cuDeviceTotalMem(&deviceProperties[deviceNum].globalMem, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].constantMem, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].numMPs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].maxRegistersPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].maxRegistersPerMP, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].pciBusId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].pciDeviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].memoryClockKhz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, deviceProperties[deviceNum].device));
        checkError(cuDeviceGetAttribute(&deviceProperties[deviceNum].memoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, deviceProperties[deviceNum].device));
        deviceProperties[deviceNum].memoryBandwidthGBs = deviceProperties[deviceNum].memoryClockKhz / 1000000.0 / 8.0 * deviceProperties[deviceNum].memoryBusWidth;

    }
}

void CudaMgr::createDeviceContexts() {
    deviceContexts.resize(deviceCount);
    for (int d = 0; d < deviceCount; ++d) {
         CUresult status = cuCtxCreate(&deviceContexts[d], 0, deviceProperties[d].device);
         if (status != CUDA_SUCCESS) {
             // this is called from destructor so we need
             // to clean up
             // destroy all contexts up to this point
             for (int destroyId = 0; destroyId <= d; ++destroyId) {
                checkError(cuCtxDestroy(deviceContexts[destroyId])); // check error here - what if it throws?
            }
            // now throw via checkError
            checkError(status);
        }
    }
}


void CudaMgr::printDeviceProperties() {
    cout << "Num devices: " << deviceCount << endl << endl;
    for (int d = 0; d < deviceCount; ++d) {
        cout << "Device: " << deviceProperties[d].device << endl;
        cout << "Compute Major: " << deviceProperties[d].computeMajor << endl;
        cout << "Compute Minor: " << deviceProperties[d].computeMinor << endl;
        cout << "PCI bus id: " << deviceProperties[d].pciBusId << endl;
        cout << "PCI deviceId id: " << deviceProperties[d].pciDeviceId << endl;
        cout << "Total Global memory: " << deviceProperties[d].globalMem / 1073741824.0 << " GB" <<  endl;
        cout << "Memory clock (khz): " << deviceProperties[d].memoryClockKhz << endl;
        cout << "Memory bandwidth: " << deviceProperties[d].memoryBandwidthGBs << " GB/sec" << endl;
        cout << endl;
    }
}

void CudaMgr::setContext(const int deviceNum) {
    //assert (deviceNum < deviceCount);
    cuCtxSetCurrent(deviceContexts[deviceNum]);
}

int8_t * CudaMgr::allocatePinnedHostMem(const size_t numBytes) {
    //setContext(deviceNum);
    void * hostPtr;
    checkError(cuMemHostAlloc(&hostPtr,numBytes,CU_MEMHOSTALLOC_PORTABLE));
    return (reinterpret_cast <int8_t *> (hostPtr));
}

int8_t * CudaMgr::allocateDeviceMem(const size_t numBytes, const int deviceNum) {
    setContext(deviceNum);
    CUdeviceptr devicePtr;
    checkError(cuMemAlloc(&devicePtr,numBytes));
    return (reinterpret_cast <int8_t *> (devicePtr));
}

void CudaMgr::freePinnedHostMem(int8_t * hostPtr) {
    checkError(cuMemFreeHost(reinterpret_cast <void *> (hostPtr)));
}

void CudaMgr::freeDeviceMem(int8_t * devicePtr) {
    checkError(cuMemFree(reinterpret_cast <CUdeviceptr> (devicePtr)));
}

void CudaMgr::copyHostToDevice(int8_t *devicePtr, const int8_t *hostPtr, const size_t numBytes, const int deviceNum) {
    setContext(deviceNum);
    checkError(cuMemcpyHtoD(reinterpret_cast <CUdeviceptr> (devicePtr),hostPtr,numBytes)); 
}

void CudaMgr::copyDeviceToHost(int8_t *hostPtr, const int8_t *devicePtr, const size_t numBytes, const int deviceNum) {
    setContext(deviceNum);
    checkError(cuMemcpyDtoH(hostPtr, reinterpret_cast <const CUdeviceptr> (devicePtr),numBytes)); 
}

void CudaMgr::copyDeviceToDevice(int8_t *destPtr, int8_t *srcPtr, const size_t numBytes, const int destDeviceNum, const int srcDeviceNum) {
    checkError(cuMemcpyPeer(reinterpret_cast<CUdeviceptr> (destPtr), deviceContexts[destDeviceNum], reinterpret_cast<CUdeviceptr> (srcPtr),deviceContexts[srcDeviceNum],numBytes)); // will we always have peer?
}



void CudaMgr::zeroDeviceMem(int8_t *devicePtr, const size_t numBytes, const int deviceNum) {
    setContext(deviceNum);
    checkError(cuMemsetD8(reinterpret_cast <CUdeviceptr> (devicePtr), 0, numBytes)); 
}



void CudaMgr::checkError(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char *errorString;
        cuGetErrorString(status,&errorString);
        // should clean up here - delete any contexts, etc.
        throw std::runtime_error(errorString);
    }
}

} // CudaMgr_Namespace

/*
int main () {
    try {
        CudaMgr_Namespace::CudaMgr cudaMgr;
        cudaMgr.printDeviceProperties();
        int numDevices = cudaMgr.getDeviceCount();
        cout << "There exists " << numDevices << " CUDA devices." << endl;
        int8_t *hostPtr, *hostPtr2, *devicePtr;
        const size_t numBytes = 1000000;
        hostPtr = cudaMgr.allocatePinnedHostMem(numBytes);
        hostPtr2 = cudaMgr.allocatePinnedHostMem(numBytes);
        devicePtr = cudaMgr.allocateDeviceMem(numBytes,0);
        for (int i = 0; i < numBytes; ++i) {
            hostPtr[i] = i % 100;
        }
        cudaMgr.copyHostToDevice(devicePtr,hostPtr,numBytes,0);
        cudaMgr.copyDeviceToHost(hostPtr2,devicePtr,numBytes,0);

        bool matchFlag = true;
        for (int i = 0; i < numBytes; ++i) {
            if (hostPtr[i] != hostPtr2[i]) {
                matchFlag = false;
                break;
            }
        }
        cout << "Match flag: " << matchFlag << endl;


        cudaMgr.setContext(0);
        cudaMgr.freeDeviceMem(devicePtr);
        cudaMgr.freePinnedHostMem(hostPtr);
    }
    catch (std::runtime_error &error) {
        cout << "Caught error: " << error.what() << endl;
    }
}
*/


