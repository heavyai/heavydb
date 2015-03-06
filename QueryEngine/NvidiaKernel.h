#ifndef QUERYENGINE_NVIDIAKERNELLAUNCH_H
#define QUERYENGINE_NVIDIAKERNELLAUNCH_H

#include <cuda.h>
#include <string>

class GpuExecutionContext {
public:
  GpuExecutionContext(const std::string& llir_module,
                      const std::string& func_name,
                      const std::string& lib_path,
                      const int device_id,
                      const void* cuda_mgr);
  ~GpuExecutionContext();
  CUfunction kernel() {
    return kernel_;
  }
private:
  CUmodule module_;
  CUfunction kernel_;
  CUlinkState link_state;
  char* ptx;
  const int device_id_;
  const void* cuda_mgr_;
};

#define checkCudaErrors(err) CHECK_EQ(err, CUDA_SUCCESS);

#endif  // QUERYENGINE_NVIDIAKERNELLAUNCH_H
