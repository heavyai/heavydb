#include "NvidiaKernel.h"

#include <glog/logging.h>


char *generatePTX(const char *ll, size_t size, const char *filename);

CUresult initCUDA(CUcontext *phContext,
                  CUdevice *phDevice,
                  CUmodule *phModule,
                  CUfunction *phKernel,
                  const char *ptx,
                  const char *libCudaDevRtName);

const char *getLibCudaDevRtName();

GpuExecutionContext::GpuExecutionContext(const std::string& llir_module)
  : hContext(nullptr)
  , hDevice(0)
  , hModule(nullptr)
  , hKernel(nullptr) {
  auto libCudaDevRtName = getLibCudaDevRtName();
  ptx = generatePTX(llir_module.c_str(), llir_module.size(), nullptr);
  CHECK(ptx);
  auto status = initCUDA(&hContext, &hDevice, &hModule, &hKernel, ptx, libCudaDevRtName);
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
