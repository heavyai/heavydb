#ifndef QUERYENGINE_NVIDIAKERNELLAUNCH_H
#define QUERYENGINE_NVIDIAKERNELLAUNCH_H

#include <cuda.h>
#include <string>

char* generatePTX(const char* ll, size_t size, const char* filename);

class GpuCompilationContext {
 public:
  GpuCompilationContext(const char* ptx,
                        const std::string& func_name,
                        const std::string& lib_path,
                        const int device_id,
                        const void* cuda_mgr,
                        const unsigned block_size_x);
  ~GpuCompilationContext();
  CUfunction kernel() { return kernel_; }

 private:
  CUmodule module_;
  CUfunction kernel_;
  CUlinkState link_state_;
  const int device_id_;
  const void* cuda_mgr_;
};

#define checkCudaErrors(err) CHECK_EQ(err, CUDA_SUCCESS);

#endif  // QUERYENGINE_NVIDIAKERNELLAUNCH_H
