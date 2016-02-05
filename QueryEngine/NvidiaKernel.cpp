#include "NvidiaKernel.h"

#include "../CudaMgr/CudaMgr.h"

#include <glog/logging.h>

GpuCompilationContext::GpuCompilationContext(const void* image,
                                             const std::string& kernel_name,
                                             const int device_id,
                                             const void* cuda_mgr,
                                             unsigned int num_options,
                                             CUjit_option* options,
                                             void** option_vals)
    : module_(nullptr), kernel_(nullptr), device_id_(device_id), cuda_mgr_(cuda_mgr) {
#ifdef HAVE_CUDA
  static_cast<const CudaMgr_Namespace::CudaMgr*>(cuda_mgr_)->setContext(device_id_);
  checkCudaErrors(cuModuleLoadDataEx(&module_, image, num_options, options, option_vals));
  CHECK(module_);
  checkCudaErrors(cuModuleGetFunction(&kernel_, module_, kernel_name.c_str()));
#endif
}

GpuCompilationContext::~GpuCompilationContext() {
#ifdef HAVE_CUDA
  static_cast<const CudaMgr_Namespace::CudaMgr*>(cuda_mgr_)->setContext(device_id_);
  auto status = cuModuleUnload(module_);
  // TODO(alex): handle this race better
  if (status == CUDA_ERROR_DEINITIALIZED) {
    return;
  }
  checkCudaErrors(status);
#endif
}
