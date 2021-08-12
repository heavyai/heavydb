#pragma once

#include "L0Mgr/L0Mgr.h"
#include "Logger/Logger.h"  // CHECK
#include "QueryEngine/CompilationContext.h"

struct L0BinResult {
  std::shared_ptr<l0::L0Device> device;
  std::shared_ptr<l0::L0Kernel> kernel;
  std::shared_ptr<l0::L0Module> module;
};

L0BinResult spv_to_bin(const std::string& spv,
                       const std::string& name,
                       const unsigned block_size,
                       const l0::L0Manager* mgr);

class L0DeviceCompilationContext {
 public:
  L0DeviceCompilationContext(std::shared_ptr<l0::L0Device> device,
                             std::shared_ptr<l0::L0Kernel> kernel,
                             std::shared_ptr<l0::L0Module> module,
                             const l0::L0Manager* l0_mgr,
                             const int device_id,
                             unsigned int num_options,
                             void** option_vals = nullptr);
  l0::L0Kernel* kernel() { return kernel_.get(); }
  l0::L0Device* device() { return device_.get(); }
  ~L0DeviceCompilationContext();

 private:
  std::shared_ptr<l0::L0Device> device_;
  std::shared_ptr<l0::L0Kernel> kernel_;
  std::shared_ptr<l0::L0Module> module_;
  const l0::L0Manager* l0_mgr_;
  const int device_id_;  // todo: remove
};

class L0CompilationContext : public CompilationContext {
 public:
  using L0DevCompilationContextPtr = std::unique_ptr<L0DeviceCompilationContext>;
  L0CompilationContext() = default;

  std::vector<l0::L0Kernel*> getNativeFunctionPointers() const {
    std::vector<l0::L0Kernel*> fn_ptrs;
    for (auto& ctx : contexts_per_device_) {
      CHECK(ctx);
      fn_ptrs.push_back(ctx->kernel());
    }
    return fn_ptrs;
  }

  l0::L0Kernel* getNativeCode(const size_t device_id) const {
    CHECK_LT(device_id, contexts_per_device_.size());
    auto device_context = contexts_per_device_[device_id].get();
    return device_context->kernel();
  }

  l0::L0Device* getDevice(const size_t device_id) const {
    CHECK_LT(device_id, contexts_per_device_.size());
    auto device_context = contexts_per_device_[device_id].get();
    return device_context->device();
  }

  void addDeviceCode(L0DevCompilationContextPtr&& device_context) {
    contexts_per_device_.push_back(move(device_context));
  }

 private:
  std::vector<L0DevCompilationContextPtr> contexts_per_device_;
};