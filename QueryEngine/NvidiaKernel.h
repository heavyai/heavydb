#ifndef QUERYENGINE_NVIDIAKERNELLAUNCH_H
#define QUERYENGINE_NVIDIAKERNELLAUNCH_H

#include "../CudaMgr/CudaMgr.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include "../Shared/nocuda.h"
#endif  // HAVE_CUDA
#include <string>
#include <vector>

struct CubinResult {
  void* cubin;
  std::vector<CUjit_option> option_keys;
  std::vector<void*> option_values;
  CUlinkState link_state;
};

CubinResult ptx_to_cubin(const std::string& ptx, const unsigned block_size, const CudaMgr_Namespace::CudaMgr* cuda_mgr);

class GpuCompilationContext {
 public:
  GpuCompilationContext(const void* image,
                        const std::string& kernel_name,
                        const int device_id,
                        const void* cuda_mgr,
                        unsigned int num_options,
                        CUjit_option* options,
                        void** option_vals);
  ~GpuCompilationContext();
  CUfunction kernel() { return kernel_; }

 private:
  CUmodule module_;
  CUfunction kernel_;
  const int device_id_;
  const void* cuda_mgr_;
};

#define checkCudaErrors(err) CHECK_EQ(err, CUDA_SUCCESS);

#endif  // QUERYENGINE_NVIDIAKERNELLAUNCH_H
