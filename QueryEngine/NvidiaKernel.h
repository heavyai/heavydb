#ifndef QUERYENGINE_NVIDIAKERNELLAUNCH_H
#define QUERYENGINE_NVIDIAKERNELLAUNCH_H

#include <cuda.h>
#include <string>

class GpuExecutionContext {
public:
  GpuExecutionContext(const std::string& llir_module, const std::string& func_name);
  ~GpuExecutionContext();
  CUfunction kernel() {
    return hKernel;
  }
private:
  CUcontext hContext;
  CUdevice hDevice;
  CUmodule hModule;
  CUfunction hKernel;
  char* ptx;
};

#endif  // QUERYENGINE_NVIDIAKERNELLAUNCH_H
