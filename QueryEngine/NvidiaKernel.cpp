#include "NvidiaKernel.h"

#include <glog/logging.h>


char *generatePTX(const char *ll, size_t size, const char *filename);

GpuExecutionContext::GpuExecutionContext(const std::string& llir_module, const std::string& func_name)
  : hModule(nullptr)
  , hKernel(nullptr) {
  ptx = generatePTX(llir_module.c_str(), llir_module.size(), nullptr);
  CHECK(ptx);
  auto status = cuModuleLoadDataEx(&hModule, ptx, 0, 0, 0);
  CHECK_EQ(status, CUDA_SUCCESS);
  CHECK(hModule);
  status = cuModuleGetFunction(&hKernel, hModule, func_name.c_str());
  CHECK_EQ(status, CUDA_SUCCESS);
}

GpuExecutionContext::~GpuExecutionContext() {
  auto status = cuModuleUnload(hModule);
  CHECK_EQ(status, CUDA_SUCCESS);
  free(ptx);
}
