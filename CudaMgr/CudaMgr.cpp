#include "CudaMgr.h"
#include <iostream>

using std::cout;
using std::endl;

namespace CudaMgr_Namespace {

CudaMgr::CudaMgr() {
    cout << "Before get device count" << endl;
    checkError(cuInit(0));
    checkError(cuDeviceGetCount(&deviceCount));
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


void CudaMgr::checkError(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char *errorString;
        cuGetErrorString(status,&errorString);
        // should clean up here - delete any contexts, etc.
        throw std::runtime_error(errorString);
    }
}

} // CudaMgr_Namespacek

int main () {
    try {
        CudaMgr_Namespace::CudaMgr cudaMgr;
        cout << "Num devices: " << cudaMgr.deviceCount << endl;
        for (int d = 0; d < cudaMgr.deviceCount; ++d) {
            cout << "Device: " << d << endl;
            cout << "Compute Major: " << cudaMgr.deviceProperties[d].computeMajor << endl;
            cout << "Compute Minor: " << cudaMgr.deviceProperties[d].computeMinor << endl;
            cout << "Total Global memory: " << cudaMgr.deviceProperties[d].globalMem << endl;
            cout << "Memory clock (khz): " << cudaMgr.deviceProperties[d].memoryClockKhz << endl;
            cout << "Memory bandwidth: " << cudaMgr.deviceProperties[d].memoryBandwidthGBs << " GB/sec" << endl;
        }
    }
    catch (std::runtime_error &error) {
        cout << "Caught error: " << error.what() << endl;
    }

}


