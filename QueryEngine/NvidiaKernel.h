#ifndef QUERYENGINE_NVIDIAKERNELLAUNCH_H
#define QUERYENGINE_NVIDIAKERNELLAUNCH_H

#include <cuda.h>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
namespace {
void fill_options(std::vector<CUjit_option>& option_keys,
                  std::vector<void*>& option_values,
                  const unsigned block_size_x) {
  option_keys.push_back(CU_JIT_LOG_VERBOSE);
  option_values.push_back(reinterpret_cast<void*>(1));
  option_keys.push_back(CU_JIT_THREADS_PER_BLOCK);
  option_values.push_back(reinterpret_cast<void*>(block_size_x));
}
}
#endif

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
