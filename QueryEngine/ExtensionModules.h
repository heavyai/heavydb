/*
    Copyright 2021 OmniSci, Inc.
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <llvm/IR/Module.h>
#include <map>
#include <memory>

enum class ExtModuleKinds {
  template_module,     // RuntimeFunctions.bc
  l0_template_module,  // RuntimeFunctionsL0.bc
  udf_cpu_module,      // Load-time UDFs for CPU execution
  udf_gpu_module,      // Load-time UDFs for GPU execution
  rt_udf_cpu_module,   // Run-time UDF/UDTFs for CPU execution
  rt_udf_gpu_module,   // Run-time UDF/UDTFs for GPU execution
  rt_libdevice_module  // math library functions for GPU execution
};

class ExtensionModuleContext {
 public:
  ExtensionModuleContext() {}

  std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& getExtensionModules() {
    // todo: thread safety?
    return extension_modules_;
  }

  const std::unique_ptr<llvm::Module>& getExtensionModule(ExtModuleKinds kind) const {
    auto it = extension_modules_.find(kind);
    if (it != extension_modules_.end()) {
      return it->second;
    }
    static const std::unique_ptr<llvm::Module> empty;
    return empty;
  }

  // Convenience functions for retrieving executor-local extension modules, thread-safe:
  const std::unique_ptr<llvm::Module>& getRTModule(bool is_l0) const {
    return getExtensionModule(is_l0 ? ExtModuleKinds::l0_template_module
                                    : ExtModuleKinds::template_module);
  }

  const std::unique_ptr<llvm::Module>& getUdfModule(bool is_gpu = false) const {
    return getExtensionModule(
        (is_gpu ? ExtModuleKinds::udf_gpu_module : ExtModuleKinds::udf_cpu_module));
  }

  // defined in Execute.cpp
  const std::unique_ptr<llvm::Module>& getRTUdfModule(bool is_gpu = false) const;

  void clear(const bool discard_runtime_modules_only) {
    if (discard_runtime_modules_only) {
      extension_modules_.erase(ExtModuleKinds::rt_udf_cpu_module);
#ifdef HAVE_CUDA
      extension_modules_.erase(ExtModuleKinds::rt_udf_gpu_module);
#endif
    } else {
      extension_modules_.clear();
    }
  }

 private:
  std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>> extension_modules_;
};
