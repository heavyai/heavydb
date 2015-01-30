#include "CudaMgr.h"
#include <iostream>

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

void CudaMgr::checkError(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char *errorString;
        cuGetErrorString(status,&errorString);
        // should clean up here - delete any contexts, etc.
        throw std::runtime_error(errorString);
    }
}

} // CudaMgr_Namespace

int main () {
    try {
        CudaMgr_Namespace::CudaMgr cudaMgr;
        cudaMgr.printDeviceProperties();
    }
    catch (std::runtime_error &error) {
        cout << "Caught error: " << error.what() << endl;
    }

}


