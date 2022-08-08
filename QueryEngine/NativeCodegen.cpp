/*
 * Copyright 2021 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "QueryEngine/Execute.h"

#if LLVM_VERSION_MAJOR < 9
static_assert(false, "LLVM Version >= 9 is required.");
#endif

#include <llvm/Analysis/ScopedNoAliasAA.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>

#ifdef ENABLE_ORCJIT
#include <llvm/ExecutionEngine/JITSymbol.h>
#else
#include <llvm/ExecutionEngine/MCJIT.h>
#endif

#include <llvm/IR/Attributes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/InferFunctionAttrs.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

#if LLVM_VERSION_MAJOR >= 11
#include <llvm/Support/Host.h>
#endif

#include "CudaMgr/CudaMgr.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Compiler/Backend.h"
#include "QueryEngine/Compiler/HelperFunctions.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/GpuSharedMemoryUtils.h"
#include "QueryEngine/LLVMFunctionAttributesUtil.h"
#include "QueryEngine/MemoryLayoutBuilder.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/QueryTemplateGenerator.h"
#include "Shared/InlineNullValues.h"
#include "Shared/MathUtils.h"
#include "StreamingTopN.h"

#include <boost/filesystem.hpp>

namespace {

/* SHOW_DEFINED(<llvm::Module instance>) prints the function names
   that are defined in the given LLVM Module instance.

   SHOW_FUNCTIONS(<llvm::Module instance>) prints the function names
   of all used functions in the given LLVM Module
   instance. Declarations are marked with `[decl]` as a name suffix.

   Useful for debugging.
*/

#define SHOW_DEFINED(MODULE)                                         \
  {                                                                  \
    std::cout << __func__ << "#" << __LINE__ << ": " #MODULE << " "; \
    ::show_defined(MODULE);                                          \
  }

#define SHOW_FUNCTIONS(MODULE)                                       \
  {                                                                  \
    std::cout << __func__ << "#" << __LINE__ << ": " #MODULE << " "; \
    ::show_functions(MODULE);                                        \
  }

template <typename T = void>
void show_defined(llvm::Module& llvm_module) {
  std::cout << "defines: ";
  for (auto& f : llvm_module.getFunctionList()) {
    if (!f.isDeclaration()) {
      std::cout << f.getName().str() << ", ";
    }
  }
  std::cout << std::endl;
}

template <typename T = void>
void show_defined(llvm::Module* llvm_module) {
  if (llvm_module == nullptr) {
    std::cout << "is null" << std::endl;
  } else {
    show_defined(*llvm_module);
  }
}

template <typename T = void>
void show_defined(std::unique_ptr<llvm::Module>& llvm_module) {
  show_defined(llvm_module.get());
}

/*
  scan_function_calls(module, defined, undefined, ignored) computes
  defined and undefined sets of function names:

  - defined functions are those that are defined in the given module

  - undefined functions are those that are called by defined functions
    but that are not defined in the given module

  - ignored functions are functions that may be undefined but will not
    be listed in the set of undefined functions.

   Useful for debugging.
*/
template <typename T = void>
void scan_function_calls(llvm::Function& F,
                         std::unordered_set<std::string>& defined,
                         std::unordered_set<std::string>& undefined,
                         const std::unordered_set<std::string>& ignored) {
  for (llvm::inst_iterator I = llvm::inst_begin(F), E = llvm::inst_end(F); I != E; ++I) {
    if (auto* CI = llvm::dyn_cast<llvm::CallInst>(&*I)) {
      auto* F2 = CI->getCalledFunction();
      if (F2 != nullptr) {
        auto F2name = F2->getName().str();
        if (F2->isDeclaration()) {
          if (F2name.rfind("__", 0) !=
                  0  // assume symbols with double underscore are defined
              && F2name.rfind("llvm.", 0) !=
                     0  // TODO: this may give false positive for NVVM intrinsics
              && ignored.find(F2name) == ignored.end()  // not in ignored list
          ) {
            undefined.emplace(F2name);
          }
        } else {
          if (defined.find(F2name) == defined.end()) {
            defined.emplace(F2name);
            scan_function_calls<T>(*F2, defined, undefined, ignored);
          }
        }
      }
    }
  }
}

template <typename T = void>
void scan_function_calls(llvm::Module& llvm_module,
                         std::unordered_set<std::string>& defined,
                         std::unordered_set<std::string>& undefined,
                         const std::unordered_set<std::string>& ignored) {
  for (auto& F : llvm_module) {
    if (!F.isDeclaration()) {
      scan_function_calls(F, defined, undefined, ignored);
    }
  }
}

template <typename T = void>
std::tuple<std::unordered_set<std::string>, std::unordered_set<std::string>>
scan_function_calls(llvm::Module& llvm_module,
                    const std::unordered_set<std::string>& ignored = {}) {
  std::unordered_set<std::string> defined, undefined;
  scan_function_calls(llvm_module, defined, undefined, ignored);
  return std::make_tuple(defined, undefined);
}

CodeCacheKey get_code_cache_key(llvm::Function* query_func, CgenState* cgen_state) {
  CodeCacheKey key{serialize_llvm_object(query_func),
                   serialize_llvm_object(cgen_state->row_func_)};
  if (cgen_state->filter_func_) {
    key.push_back(serialize_llvm_object(cgen_state->filter_func_));
  }
  for (const auto helper : cgen_state->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  return key;
}

}  // namespace

std::shared_ptr<CompilationContext> Executor::optimizeAndCodegenCPU(
    llvm::Function* query_func,
    llvm::Function* multifrag_query_func,
    std::shared_ptr<compiler::Backend> backend,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  auto key = get_code_cache_key(query_func, cgen_state_.get());
  auto cached_code = cpu_code_accessor->get_value(key);
  if (cached_code) {
    return cached_code;
  }

  std::shared_ptr<CpuCompilationContext> cpu_compilation_context =
      std::dynamic_pointer_cast<CpuCompilationContext>(
          backend->generateNativeCode(query_func, nullptr, live_funcs, co));
  cpu_compilation_context->setFunctionPointer(multifrag_query_func);
  cpu_code_accessor->put(key, cpu_compilation_context);
  return std::dynamic_pointer_cast<CompilationContext>(cpu_compilation_context);
}

void CodeGenerator::link_udf_module(const std::unique_ptr<llvm::Module>& udf_module,
                                    llvm::Module& llvm_module,
                                    CgenState* cgen_state,
                                    llvm::Linker::Flags flags) {
  auto timer = DEBUG_TIMER(__func__);
  // throw a runtime error if the target module contains functions
  // with the same name as in module of UDF functions.
  for (auto& f : *udf_module) {
    auto func = llvm_module.getFunction(f.getName());
    if (!(func == nullptr) && !f.isDeclaration() && flags == llvm::Linker::Flags::None) {
      LOG(ERROR) << "  Attempt to overwrite " << f.getName().str() << " in "
                 << llvm_module.getModuleIdentifier() << " from `"
                 << udf_module->getModuleIdentifier() << "`" << std::endl;
      throw std::runtime_error(
          "link_udf_module: *** attempt to overwrite a runtime function with a UDF "
          "function ***");
    } else {
      VLOG(1) << "  Adding " << f.getName().str() << " to "
              << llvm_module.getModuleIdentifier() << " from `"
              << udf_module->getModuleIdentifier() << "`" << std::endl;
    }
  }

  auto udf_module_copy = llvm::CloneModule(*udf_module, cgen_state->vmap_);

  udf_module_copy->setDataLayout(llvm_module.getDataLayout());
  udf_module_copy->setTargetTriple(llvm_module.getTargetTriple());

  // Initialize linker with module for RuntimeFunctions.bc
  llvm::Linker ld(llvm_module);
  bool link_error = false;

  link_error = ld.linkInModule(std::move(udf_module_copy), flags);

  if (link_error) {
    throw std::runtime_error("link_udf_module: *** error linking module ***");
  }
}

std::map<std::string, std::string> get_device_parameters(bool cpu_only) {
  std::map<std::string, std::string> result;

  result.insert(std::make_pair("cpu_name", llvm::sys::getHostCPUName()));
  result.insert(std::make_pair("cpu_triple", llvm::sys::getProcessTriple()));
  result.insert(
      std::make_pair("cpu_cores", std::to_string(llvm::sys::getHostNumPhysicalCores())));
  result.insert(std::make_pair("cpu_threads", std::to_string(cpu_threads())));

  // https://en.cppreference.com/w/cpp/language/types
  std::string sizeof_types;
  sizeof_types += "bool:" + std::to_string(sizeof(bool)) + ";";
  sizeof_types += "size_t:" + std::to_string(sizeof(size_t)) + ";";
  sizeof_types += "ssize_t:" + std::to_string(sizeof(ssize_t)) + ";";
  sizeof_types += "char:" + std::to_string(sizeof(char)) + ";";
  sizeof_types += "uchar:" + std::to_string(sizeof(unsigned char)) + ";";
  sizeof_types += "short:" + std::to_string(sizeof(short)) + ";";
  sizeof_types += "ushort:" + std::to_string(sizeof(unsigned short int)) + ";";
  sizeof_types += "int:" + std::to_string(sizeof(int)) + ";";
  sizeof_types += "uint:" + std::to_string(sizeof(unsigned int)) + ";";
  sizeof_types += "long:" + std::to_string(sizeof(long int)) + ";";
  sizeof_types += "ulong:" + std::to_string(sizeof(unsigned long int)) + ";";
  sizeof_types += "longlong:" + std::to_string(sizeof(long long int)) + ";";
  sizeof_types += "ulonglong:" + std::to_string(sizeof(unsigned long long int)) + ";";
  sizeof_types += "float:" + std::to_string(sizeof(float)) + ";";
  sizeof_types += "double:" + std::to_string(sizeof(double)) + ";";
  sizeof_types += "longdouble:" + std::to_string(sizeof(long double)) + ";";
  sizeof_types += "voidptr:" + std::to_string(sizeof(void*)) + ";";

  result.insert(std::make_pair("type_sizeof", sizeof_types));

  std::string null_values;
  null_values += "boolean1:" + std::to_string(serialized_null_value<bool>()) + ";";
  null_values += "boolean8:" + std::to_string(serialized_null_value<int8_t>()) + ";";
  null_values += "int8:" + std::to_string(serialized_null_value<int8_t>()) + ";";
  null_values += "int16:" + std::to_string(serialized_null_value<int16_t>()) + ";";
  null_values += "int32:" + std::to_string(serialized_null_value<int32_t>()) + ";";
  null_values += "int64:" + std::to_string(serialized_null_value<int64_t>()) + ";";
  null_values += "uint8:" + std::to_string(serialized_null_value<uint8_t>()) + ";";
  null_values += "uint16:" + std::to_string(serialized_null_value<uint16_t>()) + ";";
  null_values += "uint32:" + std::to_string(serialized_null_value<uint32_t>()) + ";";
  null_values += "uint64:" + std::to_string(serialized_null_value<uint64_t>()) + ";";
  null_values += "float32:" + std::to_string(serialized_null_value<float>()) + ";";
  null_values += "float64:" + std::to_string(serialized_null_value<double>()) + ";";
  null_values +=
      "Array<boolean8>:" + std::to_string(serialized_null_value<int8_t, true>()) + ";";
  null_values +=
      "Array<int8>:" + std::to_string(serialized_null_value<int8_t, true>()) + ";";
  null_values +=
      "Array<int16>:" + std::to_string(serialized_null_value<int16_t, true>()) + ";";
  null_values +=
      "Array<int32>:" + std::to_string(serialized_null_value<int32_t, true>()) + ";";
  null_values +=
      "Array<int64>:" + std::to_string(serialized_null_value<int64_t, true>()) + ";";
  null_values +=
      "Array<float32>:" + std::to_string(serialized_null_value<float, true>()) + ";";
  null_values +=
      "Array<float64>:" + std::to_string(serialized_null_value<double, true>()) + ";";

  result.insert(std::make_pair("null_values", null_values));

  llvm::StringMap<bool> cpu_features;
  if (llvm::sys::getHostCPUFeatures(cpu_features)) {
    std::string features_str = "";
    for (auto it = cpu_features.begin(); it != cpu_features.end(); ++it) {
      features_str += (it->getValue() ? " +" : " -");
      features_str += it->getKey().str();
    }
    result.insert(std::make_pair("cpu_features", features_str));
  }

  result.insert(std::make_pair("llvm_version",
                               std::to_string(LLVM_VERSION_MAJOR) + "." +
                                   std::to_string(LLVM_VERSION_MINOR) + "." +
                                   std::to_string(LLVM_VERSION_PATCH)));

#ifdef HAVE_CUDA
  if (!cpu_only) {
    int device_count = 0;
    checkCudaErrors(cuDeviceGetCount(&device_count));
    if (device_count) {
      CUdevice device{};
      char device_name[256];
      int major = 0, minor = 0;
      int driver_version;
      checkCudaErrors(cuDeviceGet(&device, 0));  // assuming homogeneous multi-GPU system
      checkCudaErrors(cuDeviceGetName(device_name, 256, device));
      checkCudaErrors(cuDeviceGetAttribute(
          &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
      checkCudaErrors(cuDeviceGetAttribute(
          &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
      checkCudaErrors(cuDriverGetVersion(&driver_version));

      result.insert(std::make_pair("gpu_name", device_name));
      result.insert(std::make_pair("gpu_count", std::to_string(device_count)));
      result.insert(std::make_pair("gpu_compute_capability",
                                   std::to_string(major) + "." + std::to_string(minor)));
      result.insert(
          std::make_pair("gpu_triple", compiler::get_gpu_target_triple_string()));
      result.insert(std::make_pair("gpu_datalayout", compiler::get_gpu_data_layout()));
      result.insert(std::make_pair("gpu_driver",
                                   "CUDA " + std::to_string(driver_version / 1000) + "." +
                                       std::to_string((driver_version % 1000) / 10)));
    }
  }
#endif

  return result;
}

std::shared_ptr<CompilationContext> Executor::optimizeAndCodegenGPU(
    llvm::Function* query_func,
    llvm::Function* multifrag_query_func,
    std::shared_ptr<compiler::Backend> backend,
    std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
#ifdef HAVE_CUDA
  auto timer = DEBUG_TIMER(__func__);

  auto key = get_code_cache_key(query_func, cgen_state_.get());
  auto cached_code = Executor::gpu_code_accessor->get_value(key);
  if (cached_code) {
    return cached_code;
  }

  std::shared_ptr<CudaCompilationContext> compilation_context;

  try {
    compilation_context = std::dynamic_pointer_cast<CudaCompilationContext>(
        backend->generateNativeCode(query_func, multifrag_query_func, live_funcs, co));

  } catch (CudaMgr_Namespace::CudaErrorException& cuda_error) {
    if (cuda_error.getStatus() == CUDA_ERROR_OUT_OF_MEMORY) {
      // Thrown if memory not able to be allocated on gpu
      // Retry once after evicting portion of code cache
      LOG(WARNING) << "Failed to allocate GPU memory for generated code. Evicting "
                   << config_->cache.gpu_fraction_code_cache_to_evict * 100.
                   << "% of GPU code cache and re-trying.";
      Executor::gpu_code_accessor->evictFractionEntries(
          config_->cache.gpu_fraction_code_cache_to_evict);

      compilation_context = std::dynamic_pointer_cast<CudaCompilationContext>(
          backend->generateNativeCode(query_func, multifrag_query_func, live_funcs, co));

    } else {
      throw;
    }
  }
  Executor::gpu_code_accessor->put(key, compilation_context);

  return std::dynamic_pointer_cast<CompilationContext>(compilation_context);
#else
  return nullptr;
#endif
}

// A small number of runtime functions don't get through CgenState::emitCall. List them
// explicitly here and always clone their implementation from the runtime module.
bool CodeGenerator::alwaysCloneRuntimeFunction(const llvm::Function* func) {
  return func->getName() == "query_stub_hoisted_literals" ||
         func->getName() == "multifrag_query_hoisted_literals" ||
         func->getName() == "query_stub" || func->getName() == "multifrag_query" ||
         func->getName() == "fixed_width_int_decode" ||
         func->getName() == "fixed_width_unsigned_decode" ||
         func->getName() == "diff_fixed_width_int_decode" ||
         func->getName() == "fixed_width_double_decode" ||
         func->getName() == "fixed_width_float_decode" ||
         func->getName() == "fixed_width_small_date_decode" ||
         func->getName() == "record_error_code" || func->getName() == "get_error_code" ||
         func->getName() == "pos_start_impl" || func->getName() == "pos_step_impl" ||
         func->getName() == "group_buff_idx_impl" ||
         func->getName() == "init_shared_mem" ||
         func->getName() == "init_shared_mem_nop" || func->getName() == "write_back_nop";
}

std::unique_ptr<llvm::Module> read_llvm_module_from_bc_file(
    const std::string& bc_filename,
    llvm::LLVMContext& context) {
  llvm::SMDiagnostic err;

  auto buffer_or_error = llvm::MemoryBuffer::getFile(bc_filename);
  CHECK(!buffer_or_error.getError()) << "bc_filename=" << bc_filename;

  llvm::MemoryBuffer* buffer = buffer_or_error.get().get();

  auto owner = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
  CHECK(!owner.takeError());
  CHECK(owner->get());
  return std::move(owner.get());
}

std::unique_ptr<llvm::Module> read_llvm_module_from_ir_file(
    const std::string& udf_ir_filename,
    llvm::LLVMContext& ctx,
    bool is_gpu = false) {
  llvm::SMDiagnostic parse_error;

  llvm::StringRef file_name_arg(udf_ir_filename);
  auto owner = llvm::parseIRFile(file_name_arg, parse_error, ctx);
  if (!owner) {
    compiler::throw_parseIR_error(parse_error, udf_ir_filename, is_gpu);
  }

  if (is_gpu) {
    llvm::Triple gpu_triple(owner->getTargetTriple());
    if (!gpu_triple.isNVPTX()) {
      LOG(WARNING)
          << "Expected triple nvptx64-nvidia-cuda for NVVM IR of loadtime UDFs but got "
          << gpu_triple.str() << ". Disabling the NVVM IR module.";
      return std::unique_ptr<llvm::Module>();
    }
  }
  return owner;
}

std::unique_ptr<llvm::Module> read_llvm_module_from_ir_string(
    const std::string& udf_ir_string,
    llvm::LLVMContext& ctx,
    bool is_gpu = false) {
  llvm::SMDiagnostic parse_error;

  auto buf = std::make_unique<llvm::MemoryBufferRef>(udf_ir_string,
                                                     "Runtime UDF/UDTF LLVM/NVVM IR");

  auto owner = llvm::parseIR(*buf, parse_error, ctx);
  if (!owner) {
    LOG(IR) << "read_llvm_module_from_ir_string:\n"
            << udf_ir_string << "\nEnd of LLVM/NVVM IR";
    compiler::throw_parseIR_error(parse_error, "", /* is_gpu= */ is_gpu);
  }

  if (is_gpu) {
    llvm::Triple gpu_triple(owner->getTargetTriple());
    if (!gpu_triple.isNVPTX()) {
      LOG(IR) << "read_llvm_module_from_ir_string:\n"
              << udf_ir_string << "\nEnd of NNVM IR";
      LOG(WARNING) << "Expected triple nvptx64-nvidia-cuda for NVVM IR but got "
                   << gpu_triple.str()
                   << ". Executing runtime UDF/UDTFs on GPU will be disabled.";
      return std::unique_ptr<llvm::Module>();
      ;
    }
  }
  return owner;
}

namespace {

void bind_pos_placeholders(const std::string& pos_fn_name,
                           const bool use_resume_param,
                           llvm::Function* query_func,
                           llvm::Module* llvm_module) {
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e;
       ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& pos_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(pos_call.getCalledFunction()->getName()) == pos_fn_name) {
      if (use_resume_param) {
        const auto error_code_arg = get_arg_by_name(query_func, "error_code");
        llvm::ReplaceInstWithInst(
            &pos_call,
            llvm::CallInst::Create(llvm_module->getFunction(pos_fn_name + "_impl"),
                                   error_code_arg));
      } else {
        llvm::ReplaceInstWithInst(
            &pos_call,
            llvm::CallInst::Create(llvm_module->getFunction(pos_fn_name + "_impl")));
      }
      break;
    }
  }
}

void set_row_func_argnames(llvm::Function* row_func,
                           const size_t in_col_count,
                           const size_t agg_col_count,
                           const bool hoist_literals) {
  auto arg_it = row_func->arg_begin();

  if (agg_col_count) {
    for (size_t i = 0; i < agg_col_count; ++i) {
      arg_it->setName("out");
      ++arg_it;
    }
  } else {
    arg_it->setName("group_by_buff");
    ++arg_it;
    arg_it->setName("varlen_output_buff");
    ++arg_it;
    arg_it->setName("crt_matched");
    ++arg_it;
    arg_it->setName("total_matched");
    ++arg_it;
    arg_it->setName("old_total_matched");
    ++arg_it;
    arg_it->setName("max_matched");
    ++arg_it;
  }

  arg_it->setName("agg_init_val");
  ++arg_it;

  arg_it->setName("pos");
  ++arg_it;

  arg_it->setName("frag_row_off");
  ++arg_it;

  arg_it->setName("num_rows_per_scan");
  ++arg_it;

  if (hoist_literals) {
    arg_it->setName("literals");
    ++arg_it;
  }

  for (size_t i = 0; i < in_col_count; ++i) {
    arg_it->setName("col_buf" + std::to_string(i));
    ++arg_it;
  }

  arg_it->setName("join_hash_tables");
}

llvm::Function* create_row_function(const size_t in_col_count,
                                    const size_t agg_col_count,
                                    const bool hoist_literals,
                                    llvm::Module* llvm_module,
                                    llvm::LLVMContext& context) {
  std::vector<llvm::Type*> row_process_arg_types;

  if (agg_col_count) {
    // output (aggregate) arguments
    for (size_t i = 0; i < agg_col_count; ++i) {
      row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    }
  } else {
    // group by buffer
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    // varlen output buffer
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    // current match count
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // total match count passed from the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // old total match count returned to the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // max matched (total number of slots in the output buffer)
    row_process_arg_types.push_back(llvm::Type::getInt32Ty(context));
  }

  // aggregate init values
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // position argument
  row_process_arg_types.push_back(llvm::Type::getInt64Ty(context));

  // fragment row offset argument
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // number of rows for each scan
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // literals buffer argument
  if (hoist_literals) {
    row_process_arg_types.push_back(llvm::Type::getInt8PtrTy(context));
  }

  // column buffer arguments
  for (size_t i = 0; i < in_col_count; ++i) {
    row_process_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context));
  }

  // join hash table argument
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // generate the function
  auto ft =
      llvm::FunctionType::get(get_int_type(32, context), row_process_arg_types, false);

  auto row_func = llvm::Function::Create(
      ft, llvm::Function::ExternalLinkage, "row_func", llvm_module);

  // set the row function argument names; for debugging purposes only
  set_row_func_argnames(row_func, in_col_count, agg_col_count, hoist_literals);

  return row_func;
}

// Iterate through multifrag_query_func, replacing calls to query_fname with query_func.
void bind_query(llvm::Function* query_func,
                const std::string& query_fname,
                llvm::Function* multifrag_query_func,
                llvm::Module* llvm_module) {
  std::vector<llvm::CallInst*> query_stubs;
  for (auto it = llvm::inst_begin(multifrag_query_func),
            e = llvm::inst_end(multifrag_query_func);
       it != e;
       ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& query_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(query_call.getCalledFunction()->getName()) == query_fname) {
      query_stubs.push_back(&query_call);
    }
  }
  for (auto& S : query_stubs) {
    std::vector<llvm::Value*> args;
    for (size_t i = 0; i < S->getNumArgOperands(); ++i) {
      args.push_back(S->getArgOperand(i));
    }
    llvm::ReplaceInstWithInst(S, llvm::CallInst::Create(query_func, args, ""));
  }
}

std::vector<std::string> get_agg_fnames(const std::vector<Analyzer::Expr*>& target_exprs,
                                        const bool is_group_by) {
  std::vector<std::string> result;
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < target_exprs.size();
       ++target_idx, ++agg_col_idx) {
    const auto target_expr = target_exprs[target_idx];
    CHECK(target_expr);
    const auto target_type_info = target_expr->get_type_info();
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    const bool is_varlen =
        (target_type_info.is_string() &&
         target_type_info.get_compression() == kENCODING_NONE) ||
        target_type_info.is_array();  // TODO: should it use is_varlen_array() ?
    if (!agg_expr || agg_expr->get_aggtype() == kSAMPLE) {
      result.emplace_back(target_type_info.is_fp() ? "agg_id_double" : "agg_id");
      if (is_varlen) {
        result.emplace_back("agg_id");
      }
      continue;
    }
    const auto agg_type = agg_expr->get_aggtype();
    const auto& agg_type_info =
        agg_type != kCOUNT ? agg_expr->get_arg()->get_type_info() : target_type_info;
    switch (agg_type) {
      case kAVG: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() &&
            !agg_type_info.is_fp()) {
          throw std::runtime_error("AVG is only valid on integer and floating point");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_sum"
                                : "agg_sum_double");
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_count"
                                : "agg_count_double");
        break;
      }
      case kMIN: {
        if (agg_type_info.is_string() || agg_type_info.is_array()) {
          throw std::runtime_error("MIN on strings or arrays types not supported yet");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_min"
                                : "agg_min_double");
        break;
      }
      case kMAX: {
        if (agg_type_info.is_string() || agg_type_info.is_array()) {
          throw std::runtime_error("MAX on strings or arrays types not supported yet");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_max"
                                : "agg_max_double");
        break;
      }
      case kSUM: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() &&
            !agg_type_info.is_fp()) {
          throw std::runtime_error("SUM is only valid on integer and floating point");
        }
        result.emplace_back((agg_type_info.is_integer() || agg_type_info.is_time())
                                ? "agg_sum"
                                : "agg_sum_double");
        break;
      }
      case kCOUNT:
        result.emplace_back(agg_expr->get_is_distinct() ? "agg_count_distinct"
                                                        : "agg_count");
        break;
      case kSINGLE_VALUE: {
        result.emplace_back(agg_type_info.is_fp() ? "agg_id_double" : "agg_id");
        break;
      }
      case kSAMPLE: {
        // Note that varlen SAMPLE arguments are handled separately above
        result.emplace_back(agg_type_info.is_fp() ? "agg_id_double" : "agg_id");
        break;
      }
      case kAPPROX_COUNT_DISTINCT:
        result.emplace_back("agg_approximate_count_distinct");
        break;
      case kAPPROX_QUANTILE:
        result.emplace_back("agg_approx_quantile");
        break;
      default:
        CHECK(false);
    }
  }
  return result;
}

}  // namespace

void Executor::addUdfIrToModule(const std::string& udf_ir_filename,
                                const bool is_cuda_ir) {
  Executor::extension_module_sources[is_cuda_ir ? ExtModuleKinds::udf_gpu_module
                                                : ExtModuleKinds::udf_cpu_module] =
      udf_ir_filename;
}

std::unordered_set<llvm::Function*> CodeGenerator::markDeadRuntimeFuncs(
    llvm::Module& llvm_module,
    const std::vector<llvm::Function*>& roots,
    const std::vector<llvm::Function*>& leaves) {
  auto timer = DEBUG_TIMER(__func__);
  std::unordered_set<llvm::Function*> live_funcs;
  live_funcs.insert(roots.begin(), roots.end());
  live_funcs.insert(leaves.begin(), leaves.end());

  if (auto F = llvm_module.getFunction("init_shared_mem_nop")) {
    live_funcs.insert(F);
  }
  if (auto F = llvm_module.getFunction("write_back_nop")) {
    live_funcs.insert(F);
  }

  for (const llvm::Function* F : roots) {
    for (const llvm::BasicBlock& BB : *F) {
      for (const llvm::Instruction& I : BB) {
        if (const llvm::CallInst* CI = llvm::dyn_cast<const llvm::CallInst>(&I)) {
          live_funcs.insert(CI->getCalledFunction());
        }
      }
    }
  }

  for (llvm::Function& F : llvm_module) {
    if (!live_funcs.count(&F) && !F.isDeclaration()) {
      F.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
  }

  return live_funcs;
}

namespace {
// searches for a particular variable within a specific basic block (or all if bb_name
// is empty)
template <typename InstType>
llvm::Value* find_variable_in_basic_block(llvm::Function* func,
                                          std::string bb_name,
                                          std::string variable_name) {
  llvm::Value* result = nullptr;
  if (func == nullptr || variable_name.empty()) {
    return result;
  }
  bool is_found = false;
  for (auto bb_it = func->begin(); bb_it != func->end() && !is_found; ++bb_it) {
    if (!bb_name.empty() && bb_it->getName() != bb_name) {
      continue;
    }
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); inst_it++) {
      if (llvm::isa<InstType>(*inst_it)) {
        if (inst_it->getName() == variable_name) {
          result = &*inst_it;
          is_found = true;
          break;
        }
      }
    }
  }
  return result;
}
};  // namespace

void Executor::createErrorCheckControlFlow(
    llvm::Function* query_func,
    bool run_with_dynamic_watchdog,
    bool run_with_allowing_runtime_interrupt,
    ExecutorDeviceType device_type,
    const std::vector<InputTableInfo>& input_table_infos) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());

  // check whether the row processing was successful; currently, it can
  // fail by running out of group by buffer slots

  if (run_with_dynamic_watchdog && run_with_allowing_runtime_interrupt) {
    // when both dynamic watchdog and runtime interrupt turns on
    // we use dynamic watchdog
    run_with_allowing_runtime_interrupt = false;
  }

  {
    // disable injecting query interrupt checker if the session info is invalid
    mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor_session_mutex_);
    if (current_query_session_.empty()) {
      run_with_allowing_runtime_interrupt = false;
    }
  }

  llvm::Value* row_count = nullptr;
  if ((run_with_dynamic_watchdog || run_with_allowing_runtime_interrupt) &&
      device_type == ExecutorDeviceType::GPU) {
    row_count =
        find_variable_in_basic_block<llvm::LoadInst>(query_func, ".entry", "row_count");
  }

  bool done_splitting = false;
  for (auto bb_it = query_func->begin(); bb_it != query_func->end() && !done_splitting;
       ++bb_it) {
    llvm::Value* pos = nullptr;
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if ((run_with_dynamic_watchdog || run_with_allowing_runtime_interrupt) &&
          llvm::isa<llvm::PHINode>(*inst_it)) {
        if (inst_it->getName() == "pos") {
          pos = &*inst_it;
        }
        continue;
      }
      if (!llvm::isa<llvm::CallInst>(*inst_it)) {
        continue;
      }
      auto& row_func_call = llvm::cast<llvm::CallInst>(*inst_it);
      if (std::string(row_func_call.getCalledFunction()->getName()) == "row_process") {
        auto next_inst_it = inst_it;
        ++next_inst_it;
        auto new_bb = bb_it->splitBasicBlock(next_inst_it);
        auto& br_instr = bb_it->back();
        llvm::IRBuilder<> ir_builder(&br_instr);
        llvm::Value* err_lv = &*inst_it;
        llvm::Value* err_lv_returned_from_row_func = nullptr;
        if (run_with_dynamic_watchdog) {
          CHECK(pos);
          llvm::Value* call_watchdog_lv = nullptr;
          if (device_type == ExecutorDeviceType::GPU) {
            // In order to make sure all threads within a block see the same barrier,
            // only those blocks whose none of their threads have experienced the
            // critical edge will go through the dynamic watchdog computation
            CHECK(row_count);
            auto crit_edge_rem =
                (blockSize() & (blockSize() - 1))
                    ? ir_builder.CreateSRem(
                          row_count,
                          cgen_state_->llInt(static_cast<int64_t>(blockSize())))
                    : ir_builder.CreateAnd(
                          row_count,
                          cgen_state_->llInt(static_cast<int64_t>(blockSize() - 1)));
            auto crit_edge_threshold = ir_builder.CreateSub(row_count, crit_edge_rem);
            crit_edge_threshold->setName("crit_edge_threshold");

            // only those threads where pos < crit_edge_threshold go through dynamic
            // watchdog call
            call_watchdog_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_SLT, pos, crit_edge_threshold);
          } else {
            // CPU path: run watchdog for every 64th row
            auto dw_predicate = ir_builder.CreateAnd(pos, uint64_t(0x3f));
            call_watchdog_lv = ir_builder.CreateICmp(
                llvm::ICmpInst::ICMP_EQ, dw_predicate, cgen_state_->llInt(int64_t(0LL)));
          }
          CHECK(call_watchdog_lv);
          auto error_check_bb = bb_it->splitBasicBlock(
              llvm::BasicBlock::iterator(br_instr), ".error_check");
          auto& watchdog_br_instr = bb_it->back();

          auto watchdog_check_bb = llvm::BasicBlock::Create(
              cgen_state_->context_, ".watchdog_check", query_func, error_check_bb);
          llvm::IRBuilder<> watchdog_ir_builder(watchdog_check_bb);
          auto detected_timeout = watchdog_ir_builder.CreateCall(
              cgen_state_->module_->getFunction("dynamic_watchdog"), {});
          auto timeout_err_lv = watchdog_ir_builder.CreateSelect(
              detected_timeout, cgen_state_->llInt(Executor::ERR_OUT_OF_TIME), err_lv);
          watchdog_ir_builder.CreateBr(error_check_bb);

          llvm::ReplaceInstWithInst(
              &watchdog_br_instr,
              llvm::BranchInst::Create(
                  watchdog_check_bb, error_check_bb, call_watchdog_lv));
          ir_builder.SetInsertPoint(&br_instr);
          auto unified_err_lv = ir_builder.CreatePHI(err_lv->getType(), 2);

          unified_err_lv->addIncoming(timeout_err_lv, watchdog_check_bb);
          unified_err_lv->addIncoming(err_lv, &*bb_it);
          err_lv = unified_err_lv;
        } else if (run_with_allowing_runtime_interrupt) {
          CHECK(pos);
          llvm::Value* call_check_interrupt_lv = nullptr;
          if (device_type == ExecutorDeviceType::GPU) {
            // approximate how many times the %pos variable
            // is increased --> the number of iteration
            // here we calculate the # bit shift by considering grid/block/fragment
            // sizes since if we use the fixed one (i.e., per 64-th increment) some CUDA
            // threads cannot enter the interrupt checking block depending on the
            // fragment size --> a thread may not take care of 64 threads if an outer
            // table is not sufficiently large, and so cannot be interrupted
            int32_t num_shift_by_gridDim = shared::getExpOfTwo(gridSize());
            int32_t num_shift_by_blockDim = shared::getExpOfTwo(blockSize());
            int64_t total_num_shift = num_shift_by_gridDim + num_shift_by_blockDim;
            uint64_t interrupt_checking_freq = 32;
            // TODO: get from ExecutionOptions
            auto freq_control_knob = config_->exec.interrupt.running_query_interrupt_freq;
            CHECK_GT(freq_control_knob, 0);
            CHECK_LE(freq_control_knob, 1.0);
            if (!input_table_infos.empty()) {
              const auto& outer_table_info = *input_table_infos.begin();
              auto num_outer_table_tuples = outer_table_info.info.getNumTuples();
              if (outer_table_info.table_id < 0) {
                auto* rs = (*outer_table_info.info.fragments.begin()).resultSet;
                CHECK(rs);
                num_outer_table_tuples = rs->entryCount();
              } else {
                auto num_frags = outer_table_info.info.fragments.size();
                if (num_frags > 0) {
                  num_outer_table_tuples =
                      outer_table_info.info.fragments.begin()->getNumTuples();
                }
              }
              if (num_outer_table_tuples > 0) {
                // gridSize * blockSize --> pos_step (idx of the next row per thread)
                // we additionally multiply two to pos_step since the number of
                // dispatched blocks are double of the gridSize
                // # tuples (of fragment) / pos_step --> maximum # increment (K)
                // also we multiply 1 / freq_control_knob to K to control the frequency
                // So, needs to check the interrupt status more frequently? make K
                // smaller
                auto max_inc = uint64_t(
                    floor(num_outer_table_tuples / (gridSize() * blockSize() * 2)));
                if (max_inc < 2) {
                  // too small `max_inc`, so this correction is necessary to make
                  // `interrupt_checking_freq` be valid (i.e., larger than zero)
                  max_inc = 2;
                }
                auto calibrated_inc = uint64_t(floor(max_inc * (1 - freq_control_knob)));
                interrupt_checking_freq =
                    uint64_t(pow(2, shared::getExpOfTwo(calibrated_inc)));
                // add the coverage when interrupt_checking_freq > K
                // if so, some threads still cannot be branched to the interrupt checker
                // so we manually use smaller but close to the max_inc as freq
                if (interrupt_checking_freq > max_inc) {
                  interrupt_checking_freq = max_inc / 2;
                }
                if (interrupt_checking_freq < 8) {
                  // such small freq incurs too frequent interrupt status checking,
                  // so we fixup to the minimum freq value at some reasonable degree
                  interrupt_checking_freq = 8;
                }
              }
            }
            VLOG(1) << "Set the running query interrupt checking frequency: "
                    << interrupt_checking_freq;
            // check the interrupt flag for every interrupt_checking_freq-th iteration
            llvm::Value* pos_shifted_per_iteration =
                ir_builder.CreateLShr(pos, cgen_state_->llInt(total_num_shift));
            auto interrupt_predicate =
                ir_builder.CreateAnd(pos_shifted_per_iteration, interrupt_checking_freq);
            call_check_interrupt_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                      interrupt_predicate,
                                      cgen_state_->llInt(int64_t(0LL)));
          } else {
            // CPU path: run interrupt checker for every 64th row
            auto interrupt_predicate = ir_builder.CreateAnd(pos, uint64_t(0x3f));
            call_check_interrupt_lv =
                ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                      interrupt_predicate,
                                      cgen_state_->llInt(int64_t(0LL)));
          }
          CHECK(call_check_interrupt_lv);
          auto error_check_bb = bb_it->splitBasicBlock(
              llvm::BasicBlock::iterator(br_instr), ".error_check");
          auto& check_interrupt_br_instr = bb_it->back();

          auto interrupt_check_bb = llvm::BasicBlock::Create(
              cgen_state_->context_, ".interrupt_check", query_func, error_check_bb);
          llvm::IRBuilder<> interrupt_checker_ir_builder(interrupt_check_bb);
          auto detected_interrupt = interrupt_checker_ir_builder.CreateCall(
              cgen_state_->module_->getFunction("check_interrupt"), {});
          auto interrupt_err_lv = interrupt_checker_ir_builder.CreateSelect(
              detected_interrupt, cgen_state_->llInt(Executor::ERR_INTERRUPTED), err_lv);
          interrupt_checker_ir_builder.CreateBr(error_check_bb);

          llvm::ReplaceInstWithInst(
              &check_interrupt_br_instr,
              llvm::BranchInst::Create(
                  interrupt_check_bb, error_check_bb, call_check_interrupt_lv));
          ir_builder.SetInsertPoint(&br_instr);
          auto unified_err_lv = ir_builder.CreatePHI(err_lv->getType(), 2);

          unified_err_lv->addIncoming(interrupt_err_lv, interrupt_check_bb);
          unified_err_lv->addIncoming(err_lv, &*bb_it);
          err_lv = unified_err_lv;
        }
        if (!err_lv_returned_from_row_func) {
          err_lv_returned_from_row_func = err_lv;
        }
        if (device_type == ExecutorDeviceType::GPU && run_with_dynamic_watchdog) {
          // let kernel execution finish as expected, regardless of the observed error,
          // unless it is from the dynamic watchdog where all threads within that block
          // return together.
          err_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ,
                                         err_lv,
                                         cgen_state_->llInt(Executor::ERR_OUT_OF_TIME));
        } else {
          err_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_NE,
                                         err_lv,
                                         cgen_state_->llInt(static_cast<int32_t>(0)));
        }
        auto error_bb = llvm::BasicBlock::Create(
            cgen_state_->context_, ".error_exit", query_func, new_bb);
        const auto error_code_arg = get_arg_by_name(query_func, "error_code");
        llvm::CallInst::Create(
            cgen_state_->module_->getFunction("record_error_code"),
            std::vector<llvm::Value*>{err_lv_returned_from_row_func, error_code_arg},
            "",
            error_bb);
        llvm::ReturnInst::Create(cgen_state_->context_, error_bb);
        llvm::ReplaceInstWithInst(&br_instr,
                                  llvm::BranchInst::Create(error_bb, new_bb, err_lv));
        done_splitting = true;
        break;
      }
    }
  }
  CHECK(done_splitting);
}

std::vector<llvm::Value*> Executor::inlineHoistedLiterals() {
  AUTOMATIC_IR_METADATA(cgen_state_.get());

  std::vector<llvm::Value*> hoisted_literals;

  // row_func_ is using literals whose defs have been hoisted up to the query_func_,
  // extend row_func_ signature to include extra args to pass these literal values.
  std::vector<llvm::Type*> row_process_arg_types;

  for (llvm::Function::arg_iterator I = cgen_state_->row_func_->arg_begin(),
                                    E = cgen_state_->row_func_->arg_end();
       I != E;
       ++I) {
    row_process_arg_types.push_back(I->getType());
  }

  for (auto& element : cgen_state_->query_func_literal_loads_) {
    for (auto value : element.second) {
      row_process_arg_types.push_back(value->getType());
    }
  }

  auto ft = llvm::FunctionType::get(
      get_int_type(32, cgen_state_->context_), row_process_arg_types, false);
  auto row_func_with_hoisted_literals =
      llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             "row_func_hoisted_literals",
                             cgen_state_->row_func_->getParent());

  auto row_func_arg_it = row_func_with_hoisted_literals->arg_begin();
  for (llvm::Function::arg_iterator I = cgen_state_->row_func_->arg_begin(),
                                    E = cgen_state_->row_func_->arg_end();
       I != E;
       ++I) {
    if (I->hasName()) {
      row_func_arg_it->setName(I->getName());
    }
    ++row_func_arg_it;
  }

  decltype(row_func_with_hoisted_literals) filter_func_with_hoisted_literals{nullptr};
  decltype(row_func_arg_it) filter_func_arg_it{nullptr};
  if (cgen_state_->filter_func_) {
    // filter_func_ is using literals whose defs have been hoisted up to the row_func_,
    // extend filter_func_ signature to include extra args to pass these literal values.
    std::vector<llvm::Type*> filter_func_arg_types;

    for (llvm::Function::arg_iterator I = cgen_state_->filter_func_->arg_begin(),
                                      E = cgen_state_->filter_func_->arg_end();
         I != E;
         ++I) {
      filter_func_arg_types.push_back(I->getType());
    }

    for (auto& element : cgen_state_->query_func_literal_loads_) {
      for (auto value : element.second) {
        filter_func_arg_types.push_back(value->getType());
      }
    }

    auto ft2 = llvm::FunctionType::get(
        get_int_type(32, cgen_state_->context_), filter_func_arg_types, false);
    filter_func_with_hoisted_literals =
        llvm::Function::Create(ft2,
                               llvm::Function::ExternalLinkage,
                               "filter_func_hoisted_literals",
                               cgen_state_->filter_func_->getParent());

    filter_func_arg_it = filter_func_with_hoisted_literals->arg_begin();
    for (llvm::Function::arg_iterator I = cgen_state_->filter_func_->arg_begin(),
                                      E = cgen_state_->filter_func_->arg_end();
         I != E;
         ++I) {
      if (I->hasName()) {
        filter_func_arg_it->setName(I->getName());
      }
      ++filter_func_arg_it;
    }
  }

  std::unordered_map<int, std::vector<llvm::Value*>>
      query_func_literal_loads_function_arguments,
      query_func_literal_loads_function_arguments2;

  for (auto& element : cgen_state_->query_func_literal_loads_) {
    std::vector<llvm::Value*> argument_values, argument_values2;

    for (auto value : element.second) {
      hoisted_literals.push_back(value);
      argument_values.push_back(&*row_func_arg_it);
      if (cgen_state_->filter_func_) {
        argument_values2.push_back(&*filter_func_arg_it);
        cgen_state_->filter_func_args_[&*row_func_arg_it] = &*filter_func_arg_it;
      }
      if (value->hasName()) {
        row_func_arg_it->setName("arg_" + value->getName());
        if (cgen_state_->filter_func_) {
          filter_func_arg_it->getContext();
          filter_func_arg_it->setName("arg_" + value->getName());
        }
      }
      ++row_func_arg_it;
      ++filter_func_arg_it;
    }

    query_func_literal_loads_function_arguments[element.first] = argument_values;
    query_func_literal_loads_function_arguments2[element.first] = argument_values2;
  }

  // copy the row_func function body over
  // see
  // https://stackoverflow.com/questions/12864106/move-function-body-avoiding-full-cloning/18751365
  row_func_with_hoisted_literals->getBasicBlockList().splice(
      row_func_with_hoisted_literals->begin(),
      cgen_state_->row_func_->getBasicBlockList());

  // also replace row_func arguments with the arguments from row_func_hoisted_literals
  for (llvm::Function::arg_iterator I = cgen_state_->row_func_->arg_begin(),
                                    E = cgen_state_->row_func_->arg_end(),
                                    I2 = row_func_with_hoisted_literals->arg_begin();
       I != E;
       ++I) {
    I->replaceAllUsesWith(&*I2);
    I2->takeName(&*I);
    cgen_state_->filter_func_args_.replace(&*I, &*I2);
    ++I2;
  }

  cgen_state_->row_func_ = row_func_with_hoisted_literals;

  // and finally replace  literal placeholders
  std::vector<llvm::Instruction*> placeholders;
  std::string prefix("__placeholder__literal_");
  for (auto it = llvm::inst_begin(row_func_with_hoisted_literals),
            e = llvm::inst_end(row_func_with_hoisted_literals);
       it != e;
       ++it) {
    if (it->hasName() && it->getName().startswith(prefix)) {
      auto offset_and_index_entry =
          cgen_state_->row_func_hoisted_literals_.find(llvm::dyn_cast<llvm::Value>(&*it));
      CHECK(offset_and_index_entry != cgen_state_->row_func_hoisted_literals_.end());

      int lit_off = offset_and_index_entry->second.offset_in_literal_buffer;
      int lit_idx = offset_and_index_entry->second.index_of_literal_load;

      it->replaceAllUsesWith(
          query_func_literal_loads_function_arguments[lit_off][lit_idx]);
      placeholders.push_back(&*it);
    }
  }
  for (auto placeholder : placeholders) {
    placeholder->removeFromParent();
  }

  if (cgen_state_->filter_func_) {
    // copy the filter_func function body over
    // see
    // https://stackoverflow.com/questions/12864106/move-function-body-avoiding-full-cloning/18751365
    filter_func_with_hoisted_literals->getBasicBlockList().splice(
        filter_func_with_hoisted_literals->begin(),
        cgen_state_->filter_func_->getBasicBlockList());

    // also replace filter_func arguments with the arguments from
    // filter_func_hoisted_literals
    for (llvm::Function::arg_iterator I = cgen_state_->filter_func_->arg_begin(),
                                      E = cgen_state_->filter_func_->arg_end(),
                                      I2 = filter_func_with_hoisted_literals->arg_begin();
         I != E;
         ++I) {
      I->replaceAllUsesWith(&*I2);
      I2->takeName(&*I);
      ++I2;
    }

    cgen_state_->filter_func_ = filter_func_with_hoisted_literals;

    // and finally replace  literal placeholders
    std::vector<llvm::Instruction*> placeholders;
    std::string prefix("__placeholder__literal_");
    for (auto it = llvm::inst_begin(filter_func_with_hoisted_literals),
              e = llvm::inst_end(filter_func_with_hoisted_literals);
         it != e;
         ++it) {
      if (it->hasName() && it->getName().startswith(prefix)) {
        auto offset_and_index_entry = cgen_state_->row_func_hoisted_literals_.find(
            llvm::dyn_cast<llvm::Value>(&*it));
        CHECK(offset_and_index_entry != cgen_state_->row_func_hoisted_literals_.end());

        int lit_off = offset_and_index_entry->second.offset_in_literal_buffer;
        int lit_idx = offset_and_index_entry->second.index_of_literal_load;

        it->replaceAllUsesWith(
            query_func_literal_loads_function_arguments2[lit_off][lit_idx]);
        placeholders.push_back(&*it);
      }
    }
    for (auto placeholder : placeholders) {
      placeholder->removeFromParent();
    }
  }

  return hoisted_literals;
}

namespace {

#ifndef NDEBUG
std::string serialize_llvm_metadata_footnotes(llvm::Function* query_func,
                                              CgenState* cgen_state) {
  std::string llvm_ir;
  std::unordered_set<llvm::MDNode*> md;

  // Loop over all instructions in the query function.
  for (auto bb_it = query_func->begin(); bb_it != query_func->end(); ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
      instr_it->getAllMetadata(imd);
      for (auto [kind, node] : imd) {
        md.insert(node);
      }
    }
  }

  // Loop over all instructions in the row function.
  for (auto bb_it = cgen_state->row_func_->begin(); bb_it != cgen_state->row_func_->end();
       ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
      instr_it->getAllMetadata(imd);
      for (auto [kind, node] : imd) {
        md.insert(node);
      }
    }
  }

  // Loop over all instructions in the filter function.
  if (cgen_state->filter_func_) {
    for (auto bb_it = cgen_state->filter_func_->begin();
         bb_it != cgen_state->filter_func_->end();
         ++bb_it) {
      for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
        llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>, 100> imd;
        instr_it->getAllMetadata(imd);
        for (auto [kind, node] : imd) {
          md.insert(node);
        }
      }
    }
  }

  // Sort the metadata by canonical number and convert to text.
  if (!md.empty()) {
    std::map<size_t, std::string> sorted_strings;
    for (auto p : md) {
      std::string str;
      llvm::raw_string_ostream os(str);
      p->print(os, cgen_state->module_, true);
      os.flush();
      auto fields = split(str, {}, 1);
      if (fields.empty() || fields[0].empty()) {
        continue;
      }
      sorted_strings.emplace(std::stoul(fields[0].substr(1)), str);
    }
    llvm_ir += "\n";
    for (auto [id, text] : sorted_strings) {
      llvm_ir += text;
      llvm_ir += "\n";
    }
  }

  return llvm_ir;
}
#endif  // NDEBUG

}  // namespace

std::tuple<CompilationResult, std::unique_ptr<QueryMemoryDescriptor>>
Executor::compileWorkUnit(const std::vector<InputTableInfo>& query_infos,
                          const RelAlgExecutionUnit& ra_exe_unit,
                          const CompilationOptions& co,
                          const ExecutionOptions& eo,
                          const GpuMgr* gpu_mgr,
                          const bool allow_lazy_fetch,
                          std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                          const size_t max_groups_buffer_entry_guess,
                          const int8_t crt_min_byte_width,
                          const bool has_cardinality_estimation,
                          DataProvider* data_provider,
                          ColumnCacheMap& column_cache) {
  auto timer = DEBUG_TIMER(__func__);

  if (co.device_type == ExecutorDeviceType::GPU) {
    if (!gpu_mgr) {
      throw QueryMustRunOnCpu();
    }
  }

#ifndef NDEBUG
  static std::uint64_t counter = 0;
  ++counter;
  VLOG(1) << "CODEGEN #" << counter << ":";
  LOG(IR) << "CODEGEN #" << counter << ":";
  LOG(PTX) << "CODEGEN #" << counter << ":";
  LOG(ASM) << "CODEGEN #" << counter << ":";
#endif

  // cgenstate_manager uses RAII pattern to manage the live time of
  // CgenState instances.
  Executor::CgenStateManager cgenstate_manager(
      *this, allow_lazy_fetch, query_infos, &ra_exe_unit);  // locks compilation mutex

  addTransientStringLiterals(ra_exe_unit, row_set_mem_owner);

  MemoryLayoutBuilder mem_layout_builder(ra_exe_unit);

  RowFuncBuilder row_func_builder(ra_exe_unit, query_infos, this);
  auto query_mem_desc = mem_layout_builder.build(
      query_infos,
      eo.allow_multifrag,
      max_groups_buffer_entry_guess,
      crt_min_byte_width,
      eo.output_columnar_hint,
      eo.just_explain,
      has_cardinality_estimation ? std::optional<int64_t>(max_groups_buffer_entry_guess)
                                 : std::nullopt,
      this,
      co.device_type);

  const bool output_columnar = query_mem_desc->didOutputColumnar();
  const auto shared_memory_size = mem_layout_builder.gpuSharedMemorySize(
      query_mem_desc.get(), gpu_mgr, this, co.device_type);
  if (shared_memory_size > 0) {
    // disable interleaved bins optimization on the GPU
    query_mem_desc->setHasInterleavedBinsOnGpu(false);
    LOG(DEBUG1) << "GPU shared memory is used for the " +
                       query_mem_desc->queryDescTypeToString() + " query(" +
                       std::to_string(shared_memory_size) + " out of " +
                       std::to_string(config_->exec.group_by.gpu_smem_threshold) +
                       " bytes).";
  }

  const GpuSharedMemoryContext gpu_smem_context(shared_memory_size);

  if (co.device_type == ExecutorDeviceType::GPU) {
    const size_t num_count_distinct_descs =
        query_mem_desc->getCountDistinctDescriptorsSize();
    for (size_t i = 0; i < num_count_distinct_descs; i++) {
      const auto& count_distinct_descriptor =
          query_mem_desc->getCountDistinctDescriptor(i);
      if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::HashSet ||
          (count_distinct_descriptor.impl_type_ != CountDistinctImplType::Invalid &&
           !co.hoist_literals)) {
        throw QueryMustRunOnCpu();
      }
    }
  }

  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  CHECK(cgen_state_->module_ == nullptr);
  cgen_state_->set_module_shallow_copy(get_rt_module(), /*always_clone=*/true);

  auto is_gpu = co.device_type == ExecutorDeviceType::GPU;
  if (is_gpu) {
    cgen_state_->module_->setDataLayout(compiler::get_gpu_data_layout());
    cgen_state_->module_->setTargetTriple(compiler::get_gpu_target_triple_string());
  }
  if (has_udf_module(/*is_gpu=*/is_gpu)) {
    CodeGenerator::link_udf_module(
        get_udf_module(/*is_gpu=*/is_gpu), *cgen_state_->module_, cgen_state_.get());
  }
  if (has_rt_udf_module(/*is_gpu=*/is_gpu)) {
    CodeGenerator::link_udf_module(
        get_rt_udf_module(/*is_gpu=*/is_gpu), *cgen_state_->module_, cgen_state_.get());
  }

  AUTOMATIC_IR_METADATA(cgen_state_.get());

  auto agg_fnames =
      get_agg_fnames(ra_exe_unit.target_exprs, !ra_exe_unit.groupby_exprs.empty());

  const auto agg_slot_count = ra_exe_unit.estimator ? size_t(1) : agg_fnames.size();

  const bool is_group_by{query_mem_desc->isGroupBy()};
  auto [query_func, row_func_call] = is_group_by
                                         ? query_group_by_template(cgen_state_->module_,
                                                                   co.hoist_literals,
                                                                   *query_mem_desc,
                                                                   co.device_type,
                                                                   ra_exe_unit.scan_limit,
                                                                   gpu_smem_context)
                                         : query_template(cgen_state_->module_,
                                                          agg_slot_count,
                                                          co.hoist_literals,
                                                          !!ra_exe_unit.estimator,
                                                          gpu_smem_context);
  bind_pos_placeholders("pos_start", true, query_func, cgen_state_->module_);
  bind_pos_placeholders("group_buff_idx", false, query_func, cgen_state_->module_);
  bind_pos_placeholders("pos_step", false, query_func, cgen_state_->module_);

  cgen_state_->query_func_ = query_func;
  cgen_state_->row_func_call_ = row_func_call;
  cgen_state_->query_func_entry_ir_builder_.SetInsertPoint(
      &query_func->getEntryBlock().front());

  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(&*fetch_bb.begin());
  auto col_heads = generate_column_heads_load(ra_exe_unit.input_col_descs.size(),
                                              query_func->args().begin(),
                                              fetch_ir_builder,
                                              cgen_state_->context_);
  CHECK_EQ(ra_exe_unit.input_col_descs.size(), col_heads.size());

  cgen_state_->row_func_ = create_row_function(ra_exe_unit.input_col_descs.size(),
                                               is_group_by ? 0 : agg_slot_count,
                                               co.hoist_literals,
                                               cgen_state_->module_,
                                               cgen_state_->context_);
  CHECK(cgen_state_->row_func_);
  cgen_state_->row_func_bb_ =
      llvm::BasicBlock::Create(cgen_state_->context_, "entry", cgen_state_->row_func_);

  if (config_->exec.codegen.enable_filter_function) {
    auto filter_func_ft =
        llvm::FunctionType::get(get_int_type(32, cgen_state_->context_), {}, false);
    cgen_state_->filter_func_ = llvm::Function::Create(filter_func_ft,
                                                       llvm::Function::ExternalLinkage,
                                                       "filter_func",
                                                       cgen_state_->module_);
    CHECK(cgen_state_->filter_func_);
    cgen_state_->filter_func_bb_ = llvm::BasicBlock::Create(
        cgen_state_->context_, "entry", cgen_state_->filter_func_);
  }

  cgen_state_->current_func_ = cgen_state_->row_func_;
  cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->row_func_bb_);

  preloadFragOffsets(ra_exe_unit.input_descs, query_infos);
  RelAlgExecutionUnit body_execution_unit = ra_exe_unit;
  const auto join_loops = buildJoinLoops(
      body_execution_unit, co, eo, query_infos, data_provider, column_cache);

  plan_state_->allocateLocalColumnIds(ra_exe_unit.input_col_descs);
  for (auto& simple_qual : ra_exe_unit.simple_quals) {
    plan_state_->addSimpleQual(simple_qual);
  }
  if (!join_loops.empty()) {
    codegenJoinLoops(join_loops,
                     body_execution_unit,
                     row_func_builder,
                     query_func,
                     cgen_state_->row_func_bb_,
                     *(query_mem_desc.get()),
                     co,
                     eo);
  } else {
    const bool can_return_error =
        compileBody(ra_exe_unit, row_func_builder, *query_mem_desc, co, gpu_smem_context);
    if (can_return_error || cgen_state_->needs_error_check_ || eo.with_dynamic_watchdog ||
        eo.allow_runtime_query_interrupt) {
      createErrorCheckControlFlow(query_func,
                                  eo.with_dynamic_watchdog,
                                  eo.allow_runtime_query_interrupt,
                                  co.device_type,
                                  row_func_builder.query_infos_);
    }
  }
  std::vector<llvm::Value*> hoisted_literals;

  if (co.hoist_literals) {
    VLOG(1) << "number of hoisted literals: "
            << cgen_state_->query_func_literal_loads_.size()
            << " / literal buffer usage: " << cgen_state_->getLiteralBufferUsage(0)
            << " bytes";
  }

  if (co.hoist_literals && !cgen_state_->query_func_literal_loads_.empty()) {
    // we have some hoisted literals...
    hoisted_literals = inlineHoistedLiterals();
  }

  // replace the row func placeholder call with the call to the actual row func
  std::vector<llvm::Value*> row_func_args;
  for (size_t i = 0; i < cgen_state_->row_func_call_->getNumArgOperands(); ++i) {
    row_func_args.push_back(cgen_state_->row_func_call_->getArgOperand(i));
  }
  row_func_args.insert(row_func_args.end(), col_heads.begin(), col_heads.end());
  row_func_args.push_back(get_arg_by_name(query_func, "join_hash_tables"));
  // push hoisted literals arguments, if any
  row_func_args.insert(
      row_func_args.end(), hoisted_literals.begin(), hoisted_literals.end());
  llvm::ReplaceInstWithInst(
      cgen_state_->row_func_call_,
      llvm::CallInst::Create(cgen_state_->row_func_, row_func_args, ""));

  // replace the filter func placeholder call with the call to the actual filter func
  if (cgen_state_->filter_func_) {
    std::vector<llvm::Value*> filter_func_args;
    for (auto arg_it = cgen_state_->filter_func_args_.begin();
         arg_it != cgen_state_->filter_func_args_.end();
         ++arg_it) {
      filter_func_args.push_back(arg_it->first);
    }
    llvm::ReplaceInstWithInst(
        cgen_state_->filter_func_call_,
        llvm::CallInst::Create(cgen_state_->filter_func_, filter_func_args, ""));
  }

  // Aggregate
  plan_state_->init_agg_vals_ = init_agg_val_vec(ra_exe_unit.target_exprs,
                                                 ra_exe_unit.quals,
                                                 *query_mem_desc,
                                                 getConfig().exec.group_by.bigint_count);

  /*
   * If we have decided to use GPU shared memory (decision is not made here), then
   * we generate proper code for extra components that it needs (buffer initialization
   * and gpu reduction from shared memory to global memory). We then replace these
   * functions into the already compiled query_func (replacing two placeholders,
   * write_back_nop and init_smem_nop). The rest of the code should be as before
   * (row_func, etc.).
   */
  if (gpu_smem_context.isSharedMemoryUsed()) {
    if (query_mem_desc->getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash) {
      GpuSharedMemCodeBuilder gpu_smem_code(
          cgen_state_->module_,
          cgen_state_->context_,
          *query_mem_desc,
          target_exprs_to_infos(ra_exe_unit.target_exprs,
                                *query_mem_desc,
                                getConfig().exec.group_by.bigint_count),
          plan_state_->init_agg_vals_,
          executor_id_,
          getConfig());
      gpu_smem_code.codegen();
      gpu_smem_code.injectFunctionsInto(query_func);

      // helper functions are used for caching purposes later
      cgen_state_->helper_functions_.push_back(gpu_smem_code.getReductionFunction());
      cgen_state_->helper_functions_.push_back(gpu_smem_code.getInitFunction());
      LOG(IR) << gpu_smem_code.toString();
    }
  }

  auto multifrag_query_func = cgen_state_->module_->getFunction(
      "multifrag_query" + std::string(co.hoist_literals ? "_hoisted_literals" : ""));
  CHECK(multifrag_query_func);

  if (co.device_type == ExecutorDeviceType::GPU && eo.allow_multifrag) {
    insertErrorCodeChecker(
        multifrag_query_func, co.hoist_literals, eo.allow_runtime_query_interrupt);
  }

  bind_query(query_func,
             "query_stub" + std::string(co.hoist_literals ? "_hoisted_literals" : ""),
             multifrag_query_func,
             cgen_state_->module_);

  std::vector<llvm::Function*> root_funcs{query_func, cgen_state_->row_func_};
  if (cgen_state_->filter_func_) {
    root_funcs.push_back(cgen_state_->filter_func_);
  }
  auto live_funcs = CodeGenerator::markDeadRuntimeFuncs(
      *cgen_state_->module_, root_funcs, {multifrag_query_func});

  // Always inline the row function and the filter function.
  // We don't want register spills in the inner loops.
  // LLVM seems to correctly free up alloca instructions
  // in these functions even when they are inlined.
  mark_function_always_inline(cgen_state_->row_func_);
  if (cgen_state_->filter_func_) {
    mark_function_always_inline(cgen_state_->filter_func_);
  }

#ifndef NDEBUG
  // Add helpful metadata to the LLVM IR for debugging.
  AUTOMATIC_IR_METADATA_DONE();
#endif

  // Serialize the important LLVM IR functions to text for SQL EXPLAIN.
  std::string llvm_ir;
  if (eo.just_explain) {
    if (co.explain_type == ExecutorExplainType::Optimized) {
#ifdef WITH_JIT_DEBUG
      throw std::runtime_error(
          "Explain optimized not available when JIT runtime debug symbols are enabled");
#else
      // Note that we don't run the NVVM reflect pass here. Use LOG(IR) to get the
      // optimized IR after NVVM reflect
      llvm::legacy::PassManager pass_manager;
      compiler::optimize_ir(query_func,
                            cgen_state_->module_,
                            pass_manager,
                            live_funcs,
                            gpu_smem_context.isSharedMemoryUsed(),
                            co);
#endif  // WITH_JIT_DEBUG
    }
    llvm_ir =
        serialize_llvm_object(multifrag_query_func) + serialize_llvm_object(query_func) +
        serialize_llvm_object(cgen_state_->row_func_) +
        (cgen_state_->filter_func_ ? serialize_llvm_object(cgen_state_->filter_func_)
                                   : "");

#ifndef NDEBUG
    llvm_ir += serialize_llvm_metadata_footnotes(query_func, cgen_state_.get());
#endif
  }

  LOG(IR) << "\n\n" << query_mem_desc->toString() << "\n";
  LOG(IR) << "IR for the "
          << (co.device_type == ExecutorDeviceType::CPU ? "CPU:\n" : "GPU:\n");
#ifdef NDEBUG
  LOG(IR) << serialize_llvm_object(query_func)
          << serialize_llvm_object(cgen_state_->row_func_)
          << (cgen_state_->filter_func_ ? serialize_llvm_object(cgen_state_->filter_func_)
                                        : "")
          << "\nEnd of IR";
#else
  LOG(IR) << serialize_llvm_object(cgen_state_->module_) << "\nEnd of IR";
#endif

  // Run some basic validation checks on the LLVM IR before code is generated below.
  compiler::verify_function_ir(cgen_state_->row_func_);
  if (cgen_state_->filter_func_) {
    compiler::verify_function_ir(cgen_state_->filter_func_);
  }

  bool row_func_not_inlined = false;
  if (is_group_by || ra_exe_unit.estimator) {
    for (auto it = llvm::inst_begin(cgen_state_->row_func_),
              e = llvm::inst_end(cgen_state_->row_func_);
         it != e;
         ++it) {
      if (llvm::isa<llvm::CallInst>(*it)) {
        auto& get_gv_call = llvm::cast<llvm::CallInst>(*it);
        if (get_gv_call.getCalledFunction()->getName() == "array_size" ||
            get_gv_call.getCalledFunction()->getName() == "linear_probabilistic_count") {
          mark_function_never_inline(cgen_state_->row_func_);
          row_func_not_inlined = true;
          break;
        }
      }
    }
  }

  GPUTarget target{gpu_mgr, blockSize(), cgen_state_.get(), row_func_not_inlined};
  auto backend = compiler::getBackend(co.device_type,
                                      get_extension_modules(),
                                      gpu_smem_context.isSharedMemoryUsed(),
                                      target);

  // Generate final native code from the LLVM IR.
  return std::make_tuple(
      CompilationResult{
          co.device_type == ExecutorDeviceType::CPU
              ? optimizeAndCodegenCPU(
                    query_func, multifrag_query_func, backend, live_funcs, co)
              : optimizeAndCodegenGPU(
                    query_func, multifrag_query_func, backend, live_funcs, co),
          cgen_state_->getLiterals(),
          output_columnar,
          llvm_ir,
          std::move(gpu_smem_context)},
      std::move(query_mem_desc));
}

void Executor::insertErrorCodeChecker(llvm::Function* query_func,
                                      bool hoist_literals,
                                      bool allow_runtime_query_interrupt) {
  auto query_stub_func_name =
      "query_stub" + std::string(hoist_literals ? "_hoisted_literals" : "");
  for (auto bb_it = query_func->begin(); bb_it != query_func->end(); ++bb_it) {
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if (!llvm::isa<llvm::CallInst>(*inst_it)) {
        continue;
      }
      auto& row_func_call = llvm::cast<llvm::CallInst>(*inst_it);
      if (std::string(row_func_call.getCalledFunction()->getName()) ==
          query_stub_func_name) {
        auto next_inst_it = inst_it;
        ++next_inst_it;
        auto new_bb = bb_it->splitBasicBlock(next_inst_it);
        auto& br_instr = bb_it->back();
        llvm::IRBuilder<> ir_builder(&br_instr);
        llvm::Value* err_lv = &*inst_it;
        auto error_check_bb =
            bb_it->splitBasicBlock(llvm::BasicBlock::iterator(br_instr), ".error_check");
        llvm::Value* error_code_arg = nullptr;
        auto arg_cnt = 0;
        for (auto arg_it = query_func->arg_begin(); arg_it != query_func->arg_end();
             arg_it++, ++arg_cnt) {
          // since multi_frag_* func has anonymous arguments so we use arg_offset
          // explicitly to capture "error_code" argument in the func's argument list
          if (hoist_literals) {
            if (arg_cnt == 9) {
              error_code_arg = &*arg_it;
              break;
            }
          } else {
            if (arg_cnt == 8) {
              error_code_arg = &*arg_it;
              break;
            }
          }
        }
        CHECK(error_code_arg);
        llvm::Value* err_code = nullptr;
        if (allow_runtime_query_interrupt) {
          // decide the final error code with a consideration of interrupt status
          auto& check_interrupt_br_instr = bb_it->back();
          auto interrupt_check_bb = llvm::BasicBlock::Create(
              cgen_state_->context_, ".interrupt_check", query_func, error_check_bb);
          llvm::IRBuilder<> interrupt_checker_ir_builder(interrupt_check_bb);
          auto detected_interrupt = interrupt_checker_ir_builder.CreateCall(
              cgen_state_->module_->getFunction("check_interrupt"), {});
          auto detected_error = interrupt_checker_ir_builder.CreateCall(
              cgen_state_->module_->getFunction("get_error_code"),
              std::vector<llvm::Value*>{error_code_arg});
          err_code = interrupt_checker_ir_builder.CreateSelect(
              detected_interrupt,
              cgen_state_->llInt(Executor::ERR_INTERRUPTED),
              detected_error);
          interrupt_checker_ir_builder.CreateBr(error_check_bb);
          llvm::ReplaceInstWithInst(&check_interrupt_br_instr,
                                    llvm::BranchInst::Create(interrupt_check_bb));
          ir_builder.SetInsertPoint(&br_instr);
        } else {
          // uses error code returned from row_func and skip to check interrupt status
          ir_builder.SetInsertPoint(&br_instr);
          err_code =
              ir_builder.CreateCall(cgen_state_->module_->getFunction("get_error_code"),
                                    std::vector<llvm::Value*>{error_code_arg});
        }
        err_lv = ir_builder.CreateICmp(
            llvm::ICmpInst::ICMP_NE, err_code, cgen_state_->llInt(0));
        auto error_bb = llvm::BasicBlock::Create(
            cgen_state_->context_, ".error_exit", query_func, new_bb);
        llvm::CallInst::Create(cgen_state_->module_->getFunction("record_error_code"),
                               std::vector<llvm::Value*>{err_code, error_code_arg},
                               "",
                               error_bb);
        llvm::ReturnInst::Create(cgen_state_->context_, error_bb);
        llvm::ReplaceInstWithInst(&br_instr,
                                  llvm::BranchInst::Create(error_bb, new_bb, err_lv));
        break;
      }
    }
  }
}

bool Executor::compileBody(const RelAlgExecutionUnit& ra_exe_unit,
                           RowFuncBuilder& row_func_builder,
                           QueryMemoryDescriptor& query_mem_desc,
                           const CompilationOptions& co,
                           const GpuSharedMemoryContext& gpu_smem_context) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());

  // Switch the code generation into a separate filter function if enabled.
  // Note that accesses to function arguments are still codegenned from the
  // row function's arguments, then later automatically forwarded and
  // remapped into filter function arguments by redeclareFilterFunction().
  cgen_state_->row_func_bb_ = cgen_state_->ir_builder_.GetInsertBlock();
  llvm::Value* loop_done{nullptr};
  std::unique_ptr<Executor::FetchCacheAnchor> fetch_cache_anchor;
  if (cgen_state_->filter_func_) {
    if (cgen_state_->row_func_bb_->getName() == "loop_body") {
      auto row_func_entry_bb = &cgen_state_->row_func_->getEntryBlock();
      cgen_state_->ir_builder_.SetInsertPoint(row_func_entry_bb,
                                              row_func_entry_bb->begin());
      loop_done = cgen_state_->ir_builder_.CreateAlloca(
          get_int_type(1, cgen_state_->context_), nullptr, "loop_done");
      cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->row_func_bb_);
      cgen_state_->ir_builder_.CreateStore(cgen_state_->llBool(true), loop_done);
    }
    cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->filter_func_bb_);
    cgen_state_->current_func_ = cgen_state_->filter_func_;
    fetch_cache_anchor = std::make_unique<Executor::FetchCacheAnchor>(cgen_state_.get());
  }

  // generate the code for the filter
  std::vector<Analyzer::Expr*> primary_quals;
  std::vector<Analyzer::Expr*> deferred_quals;
  bool short_circuited = CodeGenerator::prioritizeQuals(
      ra_exe_unit, primary_quals, deferred_quals, plan_state_->hoisted_filters_);
  if (short_circuited) {
    VLOG(1) << "Prioritized " << std::to_string(primary_quals.size()) << " quals, "
            << "short-circuited and deferred " << std::to_string(deferred_quals.size())
            << " quals";
  }
  llvm::Value* filter_lv = cgen_state_->llBool(true);
  CodeGenerator code_generator(this);
  for (auto expr : primary_quals) {
    // Generate the filter for primary quals
    auto cond = code_generator.toBool(code_generator.codegen(expr, true, co).front());
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, cond);
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));
  llvm::BasicBlock* sc_false{nullptr};
  if (!deferred_quals.empty()) {
    auto sc_true = llvm::BasicBlock::Create(
        cgen_state_->context_, "sc_true", cgen_state_->current_func_);
    sc_false = llvm::BasicBlock::Create(
        cgen_state_->context_, "sc_false", cgen_state_->current_func_);
    cgen_state_->ir_builder_.CreateCondBr(filter_lv, sc_true, sc_false);
    cgen_state_->ir_builder_.SetInsertPoint(sc_false);
    if (ra_exe_unit.join_quals.empty()) {
      cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(int32_t(0)));
    }
    cgen_state_->ir_builder_.SetInsertPoint(sc_true);
    filter_lv = cgen_state_->llBool(true);
  }
  for (auto expr : deferred_quals) {
    filter_lv = cgen_state_->ir_builder_.CreateAnd(
        filter_lv, code_generator.toBool(code_generator.codegen(expr, true, co).front()));
  }

  CHECK(filter_lv->getType()->isIntegerTy(1));
  auto ret =
      row_func_builder.codegen(filter_lv, sc_false, query_mem_desc, co, gpu_smem_context);

  // Switch the code generation back to the row function if a filter
  // function was enabled.
  if (cgen_state_->filter_func_) {
    if (cgen_state_->row_func_bb_->getName() == "loop_body") {
      cgen_state_->ir_builder_.CreateStore(cgen_state_->llBool(false), loop_done);
      cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt<int32_t>(0));
    }

    cgen_state_->ir_builder_.SetInsertPoint(cgen_state_->row_func_bb_);
    cgen_state_->current_func_ = cgen_state_->row_func_;
    cgen_state_->filter_func_call_ =
        cgen_state_->ir_builder_.CreateCall(cgen_state_->filter_func_, {});

    // Create real filter function declaration after placeholder call
    // is emitted.
    redeclareFilterFunction();

    if (cgen_state_->row_func_bb_->getName() == "loop_body") {
      auto loop_done_true = llvm::BasicBlock::Create(
          cgen_state_->context_, "loop_done_true", cgen_state_->row_func_);
      auto loop_done_false = llvm::BasicBlock::Create(
          cgen_state_->context_, "loop_done_false", cgen_state_->row_func_);
      auto loop_done_flag = cgen_state_->ir_builder_.CreateLoad(
          loop_done->getType()->getPointerElementType(), loop_done);
      cgen_state_->ir_builder_.CreateCondBr(
          loop_done_flag, loop_done_true, loop_done_false);
      cgen_state_->ir_builder_.SetInsertPoint(loop_done_true);
      cgen_state_->ir_builder_.CreateRet(cgen_state_->filter_func_call_);
      cgen_state_->ir_builder_.SetInsertPoint(loop_done_false);
    } else {
      cgen_state_->ir_builder_.CreateRet(cgen_state_->filter_func_call_);
    }
  }
  return ret;
}

std::vector<llvm::Value*> generate_column_heads_load(const int num_columns,
                                                     llvm::Value* byte_stream_arg,
                                                     llvm::IRBuilder<>& ir_builder,
                                                     llvm::LLVMContext& ctx) {
  CHECK(byte_stream_arg);
  const auto max_col_local_id = num_columns - 1;

  std::vector<llvm::Value*> col_heads;
  for (int col_id = 0; col_id <= max_col_local_id; ++col_id) {
    auto* gep = ir_builder.CreateGEP(
        byte_stream_arg->getType()->getScalarType()->getPointerElementType(),
        byte_stream_arg,
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), col_id));
    col_heads.emplace_back(
        ir_builder.CreateLoad(gep->getType()->getPointerElementType(), gep));
  }
  return col_heads;
}
