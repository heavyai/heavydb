#include "NvidiaKernel.h"

#include <glog/logging.h>


char *generatePTX(const char *ll, size_t size, const char *filename);
CUdevice cudaDeviceInit();

namespace {

CUresult initCUDA(CUcontext *phContext,
    CUdevice *phDevice,
    CUmodule *phModule,
    CUfunction *phKernel,
    const char *ptx,
    const char* func_name) {
  // Initialize
  *phDevice = cudaDeviceInit();
  // Create context on the device
  auto status = cuCtxCreate(phContext, 0, *phDevice);
  CHECK_EQ(status, CUDA_SUCCESS);
  status = cuModuleLoadDataEx(phModule, ptx, 0, 0, 0);
  // Locate the kernel entry poin
  status = cuModuleGetFunction(phKernel, *phModule, func_name);
  CHECK_EQ(status, CUDA_SUCCESS);

  return CUDA_SUCCESS;
}

}

GpuExecutionContext::GpuExecutionContext(const std::string& llir_module, const std::string& func_name)
  : hContext(nullptr)
  , hDevice(0)
  , hModule(nullptr)
  , hKernel(nullptr) {
  ptx = generatePTX(llir_module.c_str(), llir_module.size(), nullptr);
  CHECK(ptx);
  auto status = initCUDA(&hContext, &hDevice, &hModule, &hKernel, ptx, func_name.c_str());
  CHECK_EQ(status, CUDA_SUCCESS);
  CHECK(hModule);
  CHECK(hContext);
}

GpuExecutionContext::~GpuExecutionContext() {
  auto status = cuModuleUnload(hModule);
  CHECK_EQ(status, CUDA_SUCCESS);
  status = cuCtxDestroy(hContext);
  CHECK_EQ(status, CUDA_SUCCESS);
  free(ptx);
}
