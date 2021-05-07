#pragma once

#include "L0Mgr/L0Mgr.h"
#include "QueryEngine/CompilationContext.h"

struct L0BinResult {
  void* l0bin{nullptr};
  size_t size{0};
};

L0BinResult spv_to_bin(const std::string& spv,
                       const unsigned block_size,
                       const l0::L0Manager* mgr);

class L0DeviceCompilationContext {
 public:
  L0DeviceCompilationContext(const void* image,
                             const size_t image_size,
                             const std::string& kernel_name,
                             const int device_id,
                             unsigned int num_options,
                             void** option_vals);
  ~L0DeviceCompilationContext();

 private:
  const int device_id_;
  const l0::L0Manager* l0_mgr_;
};

class L0CompilationContext : public CompilationContext {
 public:
  using L0DevCompilationContextPtr = std::unique_ptr<L0DeviceCompilationContext>;
  L0CompilationContext() = default;

  std::vector<void*> getNativeFunctionPointers() const {
    std::vector<void*> fn_ptrs;
    return fn_ptrs;
  }

  void addDeviceCode(L0DevCompilationContextPtr&& device_context) {
    contexts_per_device_.push_back(move(device_context));
  }

 private:
  std::vector<L0DevCompilationContextPtr> contexts_per_device_;
};