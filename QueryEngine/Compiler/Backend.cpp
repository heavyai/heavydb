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

#include "Backend.h"
#include "HelperFunctions.h"

#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ExecutionEngineWrapper.h"

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#ifdef HAVE_L0
#include "LLVMSPIRVLib/LLVMSPIRVLib.h"
#endif

#ifdef ENABLE_ORCJIT
#include <llvm/ExecutionEngine/JITSymbol.h>
#else
#include <llvm/ExecutionEngine/MCJIT.h>
#endif

#ifdef ENABLE_ORCJIT
ORCJITExecutionEngineWrapper::ORCJITExecutionEngineWrapper() {}

#else  // MCJIT
MCJITExecutionEngineWrapper::MCJITExecutionEngineWrapper() {}

MCJITExecutionEngineWrapper::MCJITExecutionEngineWrapper(
    llvm::ExecutionEngine* execution_engine,
    const CompilationOptions& co)
    : execution_engine_(execution_engine) {
  if (execution_engine_) {
    if (co.register_intel_jit_listener) {
#ifdef ENABLE_INTEL_JIT_LISTENER
      intel_jit_listener_.reset(llvm::JITEventListener::createIntelJITEventListener());
      CHECK(intel_jit_listener_);
      execution_engine_->RegisterJITEventListener(intel_jit_listener_.get());
      LOG(INFO) << "Registered IntelJITEventListener";
#else
      LOG(WARNING) << "This build is not Intel JIT Listener enabled. Ignoring Intel JIT "
                      "listener configuration parameter.";
#endif  // ENABLE_INTEL_JIT_LISTENER
    }
  }
}

MCJITExecutionEngineWrapper& MCJITExecutionEngineWrapper::operator=(
    llvm::ExecutionEngine* execution_engine) {
  execution_engine_.reset(execution_engine);
  intel_jit_listener_ = nullptr;
  return *this;
}
#endif
namespace compiler {

static llvm::sys::Mutex g_ee_create_mutex;
namespace {
void throw_parseIR_error(const llvm::SMDiagnostic& parse_error,
                         std::string src = "",
                         const bool is_gpu = false) {
  std::string excname = (is_gpu ? "NVVM IR ParseError: " : "LLVM IR ParseError: ");
  llvm::raw_string_ostream ss(excname);
  parse_error.print(src.c_str(), ss, false, false);
  throw ParseIRError(ss.str());
}

std::string assemblyForCPU(ExecutionEngineWrapper& execution_engine,
                           llvm::Module* llvm_module) {
#ifndef ENABLE_ORCJIT
  llvm::legacy::PassManager pass_manager;
  auto cpu_target_machine = execution_engine->getTargetMachine();
  CHECK(cpu_target_machine);
  llvm::SmallString<256> code_str;
  llvm::raw_svector_ostream os(code_str);
#if LLVM_VERSION_MAJOR >= 10
  cpu_target_machine->addPassesToEmitFile(
      pass_manager, os, nullptr, llvm::CGFT_AssemblyFile);
#else
  cpu_target_machine->addPassesToEmitFile(
      pass_manager, os, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);
#endif
  pass_manager.run(*llvm_module);
  return "Assembly for the CPU:\n" + std::string(code_str.str()) + "\nEnd of assembly";
#else   // ORCJIT
  LOG(FATAL) << "Assembly logger not yet supported for ORCJIT.";
  return "";
#endif  // !ENABLE_ORCJIT
}

#ifndef ENABLE_ORCJIT

std::shared_ptr<CpuCompilationContext> create_execution_engine(
    llvm::Module* llvm_module,
    llvm::EngineBuilder& eb,
    const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  // Avoids data race in
  // llvm::sys::DynamicLibrary::getPermanentLibrary and
  // GDBJITRegistrationListener::notifyObjectLoaded while creating a
  // new ExecutionEngine instance. Unfortunately we have to use global
  // mutex here.
  std::lock_guard<llvm::sys::Mutex> lock(g_ee_create_mutex);
  ExecutionEngineWrapper execution_engine(eb.create(), co);
  CHECK(execution_engine.exists());
  // Force the module data layout to match the layout for the selected target
  llvm_module->setDataLayout(execution_engine->getDataLayout());

  LOG(ASM) << assemblyForCPU(execution_engine, llvm_module);

  execution_engine->finalizeObject();
  return std::make_shared<CpuCompilationContext>(std::move(execution_engine));
}

#endif

}  // namespace

std::shared_ptr<CompilationContext> CPUBackend::generateNativeCode(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  return std::dynamic_pointer_cast<CpuCompilationContext>(
      CPUBackend::generateNativeCPUCode(func, live_funcs, co));
}

std::shared_ptr<CpuCompilationContext> CPUBackend::generateNativeCPUCode(
    llvm::Function* func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  llvm::Module* llvm_module = func->getParent();
  // run optimizations
#ifndef WITH_JIT_DEBUG
  llvm::legacy::PassManager pass_manager;
  compiler::optimize_ir(
      func, llvm_module, pass_manager, live_funcs, /*is_gpu_smem_used=*/false, co);
#endif  // WITH_JIT_DEBUG

  auto init_err = llvm::InitializeNativeTarget();
  CHECK(!init_err);

#ifndef ENABLE_ORCJIT
  llvm::InitializeAllTargetMCs();
#endif
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  std::string err_str;
  std::unique_ptr<llvm::Module> owner(llvm_module);

  llvm::TargetOptions to;
  to.EnableFastISel = true;

#ifdef ENABLE_ORCJIT

  auto llvm_err_to_str = [](const llvm::Error& err) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << err;
    return msg;
  };

#if LLVM_VERSION_MAJOR > 12
  auto self_epc = llvm::cantFail(llvm::orc::SelfExecutorProcessControl::Create());
  auto execution_session =
      std::make_unique<llvm::orc::ExecutionSession>(std::move(self_epc));
#else
  auto execution_session = std::make_unique<llvm::orc::ExecutionSession>();
#endif

  auto target_machine_builder_or_error = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!target_machine_builder_or_error) {
    LOG(FATAL) << "Failed to initialize JITTargetMachineBuilder: "
               << llvm_err_to_str(target_machine_builder_or_error.takeError());
  }
  llvm::orc::JITTargetMachineBuilder target_machine_builder =
      std::move(*target_machine_builder_or_error);
  target_machine_builder.getOptions().EnableFastISel = true;

  if (co.opt_level == ExecutorOptLevel::ReductionJIT) {
    target_machine_builder.setCodeGenOptLevel(llvm::CodeGenOpt::None);
  }

  auto data_layout_or_err = target_machine_builder.getDefaultDataLayoutForTarget();
  if (!data_layout_or_err) {
    LOG(FATAL) << "Failed to initialize data layout: "
               << llvm_err_to_str(data_layout_or_err.takeError());
  }
  std::unique_ptr<llvm::DataLayout> data_layout =
      std::make_unique<llvm::DataLayout>(std::move(*data_layout_or_err));

  ExecutionEngineWrapper execution_engine(std::move(execution_session),
                                          std::move(target_machine_builder),
                                          std::move(data_layout));
  execution_engine.addModule(std::move(owner));
  return std::make_shared<CpuCompilationContext>(std::move(execution_engine));
#else

  llvm::EngineBuilder eb(std::move(owner));
  eb.setErrorStr(&err_str);
  eb.setEngineKind(llvm::EngineKind::JIT);
  eb.setTargetOptions(to);
  if (co.opt_level == ExecutorOptLevel::ReductionJIT) {
    eb.setOptLevel(llvm::CodeGenOpt::None);
  }

  return create_execution_engine(llvm_module, eb, co);
#endif
}

std::unique_ptr<llvm::TargetMachine> CUDABackend::initializeNVPTXBackend(
    const CudaMgr_Namespace::NvidiaDeviceArch arch) {
  auto timer = DEBUG_TIMER(__func__);

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  std::string err;
  auto target = llvm::TargetRegistry::lookupTarget("nvptx64", err);
  if (!target) {
    LOG(FATAL) << err;
  }

  return std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine("nvptx64-nvidia-cuda",
                                  CudaMgr_Namespace::CudaMgr::deviceArchToSM(arch),
                                  "",
                                  llvm::TargetOptions(),
                                  llvm::Reloc::Static));
}

CUDABackend::CUDABackend(
    const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
    bool is_gpu_smem_used,
    GPUTarget& gpu_target)
    : exts_(exts), is_gpu_smem_used_(is_gpu_smem_used), gpu_target_(gpu_target) {
  CHECK(gpu_target_.gpu_mgr);
  auto cuda_mgr = dynamic_cast<const CudaMgr_Namespace::CudaMgr*>(gpu_target_.gpu_mgr);
  const auto arch = cuda_mgr->getDeviceArch();
  nvptx_target_machine_ = initializeNVPTXBackend(arch);
}

std::shared_ptr<CompilationContext> CUDABackend::generateNativeCode(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  return std::dynamic_pointer_cast<CudaCompilationContext>(
      generateNativeGPUCode(exts_,
                            func,
                            wrapper_func,
                            live_funcs,
                            is_gpu_smem_used_,
                            co,
                            gpu_target_,
                            nvptx_target_machine_.get()));
}

std::string CUDABackend::generatePTX(const std::string& cuda_llir,
                                     llvm::TargetMachine* nvptx_target_machine,
                                     llvm::LLVMContext& context) {
  auto timer = DEBUG_TIMER(__func__);
  auto mem_buff = llvm::MemoryBuffer::getMemBuffer(cuda_llir, "", false);

  llvm::SMDiagnostic parse_error;

  auto llvm_module = llvm::parseIR(mem_buff->getMemBufferRef(), parse_error, context);
  if (!llvm_module) {
    LOG(IR) << "CodeGenerator::generatePTX:NVVM IR:\n" << cuda_llir << "\nEnd of NNVM IR";
    compiler::throw_parseIR_error(parse_error, "generatePTX", /* is_gpu= */ true);
  }

  llvm::SmallString<256> code_str;
  llvm::raw_svector_ostream formatted_os(code_str);
  CHECK(nvptx_target_machine);
  {
    llvm::legacy::PassManager ptxgen_pm;
    llvm_module->setDataLayout(nvptx_target_machine->createDataLayout());

#if LLVM_VERSION_MAJOR >= 10
    nvptx_target_machine->addPassesToEmitFile(
        ptxgen_pm, formatted_os, nullptr, llvm::CGFT_AssemblyFile);
#else
    nvptx_target_machine->addPassesToEmitFile(
        ptxgen_pm, formatted_os, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);
#endif
    ptxgen_pm.run(*llvm_module);
  }

#if LLVM_VERSION_MAJOR >= 11
  return std::string(code_str);
#else
  return code_str.str();
#endif
}

namespace {

#ifdef HAVE_CUDA
std::unordered_set<llvm::Function*> findAliveRuntimeFuncs(
    llvm::Module& llvm_module,
    const std::vector<llvm::Function*>& roots) {
  std::queue<llvm::Function*> queue;
  std::unordered_set<llvm::Function*> visited;
  for (llvm::Function* F : roots) {
    queue.push(F);
  }

  while (!queue.empty()) {
    llvm::Function* F = queue.front();
    queue.pop();
    if (visited.find(F) != visited.end()) {
      continue;
    }
    visited.insert(F);

    for (llvm::inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (llvm::CallInst* CI = llvm::dyn_cast<llvm::CallInst>(&*I)) {
        if (CI->isInlineAsm())  // libdevice calls inline assembly code
          continue;
        llvm::Function* called = CI->getCalledFunction();
        if (!called || visited.find(called) != visited.end()) {
          continue;
        }
        queue.push(called);
      }
    }
  }
  return visited;
}

// check if linking with libdevice is required
// libdevice functions have a __nv_* prefix
bool check_module_requires_libdevice(llvm::Module* llvm_module) {
  auto timer = DEBUG_TIMER(__func__);
  for (llvm::Function& F : *llvm_module) {
    if (F.hasName() && F.getName().startswith("__nv_")) {
      LOG(INFO) << "Module requires linking with libdevice: " << std::string(F.getName());
      return true;
    }
  }
  LOG(DEBUG1) << "module does not require linking against libdevice";
  return false;
}

// Adds the missing intrinsics declarations to the given module
void add_intrinsics_to_module(llvm::Module* llvm_module) {
  for (llvm::Function& F : *llvm_module) {
    for (llvm::Instruction& I : instructions(F)) {
      if (llvm::IntrinsicInst* ii = llvm::dyn_cast<llvm::IntrinsicInst>(&I)) {
        if (llvm::Intrinsic::isOverloaded(ii->getIntrinsicID())) {
          llvm::Type* Tys[] = {ii->getFunctionType()->getReturnType()};
          llvm::Function& decl_fn =
              *llvm::Intrinsic::getDeclaration(llvm_module, ii->getIntrinsicID(), Tys);
          ii->setCalledFunction(&decl_fn);
        } else {
          // inserts the declaration into the module if not present
          llvm::Intrinsic::getDeclaration(llvm_module, ii->getIntrinsicID());
        }
      }
    }
  }
}

std::string extension_function_decls(const std::unordered_set<std::string>& udf_decls) {
  const auto decls =
      ExtensionFunctionsWhitelist::getLLVMDeclarations(udf_decls, /*is_gpu=*/true);
  return boost::algorithm::join(decls, "\n");
}

void legalize_nvvm_ir(llvm::Function* query_func) {
  // optimizations might add attributes to the function
  // and NVPTX doesn't understand all of them; play it
  // safe and clear all attributes
  clear_function_attributes(query_func);
  verify_function_ir(query_func);

  std::vector<llvm::Instruction*> stackrestore_intrinsics;
  std::vector<llvm::Instruction*> stacksave_intrinsics;
  std::vector<llvm::Instruction*> lifetime;
  for (auto& BB : *query_func) {
    for (llvm::Instruction& I : BB) {
      if (const llvm::IntrinsicInst* II = llvm::dyn_cast<llvm::IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == llvm::Intrinsic::stacksave) {
          stacksave_intrinsics.push_back(&I);
        } else if (II->getIntrinsicID() == llvm::Intrinsic::stackrestore) {
          stackrestore_intrinsics.push_back(&I);
        } else if (II->getIntrinsicID() == llvm::Intrinsic::lifetime_start ||
                   II->getIntrinsicID() == llvm::Intrinsic::lifetime_end) {
          lifetime.push_back(&I);
        }
      }
    }
  }

  // stacksave and stackrestore intrinsics appear together, and
  // stackrestore uses stacksaved result as its argument
  // so it should be removed first.
  for (auto& II : stackrestore_intrinsics) {
    II->eraseFromParent();
  }
  for (auto& II : stacksave_intrinsics) {
    II->eraseFromParent();
  }
  // Remove lifetime intrinsics as well. NVPTX don't like them
  for (auto& II : lifetime) {
    II->eraseFromParent();
  }
}
#endif  // HAVE_CUDA

std::string cpp_to_llvm_name(const std::string& s) {
  if (s == "int8_t") {
    return "i8";
  }
  if (s == "int16_t") {
    return "i16";
  }
  if (s == "int32_t") {
    return "i32";
  }
  if (s == "int64_t") {
    return "i64";
  }
  CHECK(s == "float" || s == "double");
  return s;
}

std::string gen_array_any_all_sigs() {
  std::string result;
  for (const std::string any_or_all : {"any", "all"}) {
    for (const std::string elem_type :
         {"int8_t", "int16_t", "int32_t", "int64_t", "float", "double"}) {
      for (const std::string needle_type :
           {"int8_t", "int16_t", "int32_t", "int64_t", "float", "double"}) {
        for (const std::string op_name : {"eq", "ne", "lt", "le", "gt", "ge"}) {
          result += ("declare i1 @array_" + any_or_all + "_" + op_name + "_" + elem_type +
                     "_" + needle_type + "(i8*, i64, " + cpp_to_llvm_name(needle_type) +
                     ", " + cpp_to_llvm_name(elem_type) + ");\n");
        }
      }
    }
  }
  return result;
}

std::string gen_translate_null_key_sigs() {
  std::string result;
  for (const std::string key_type : {"int8_t", "int16_t", "int32_t", "int64_t"}) {
    const auto key_llvm_type = cpp_to_llvm_name(key_type);
    result += "declare i64 @translate_null_key_" + key_type + "(" + key_llvm_type + ", " +
              key_llvm_type + ", i64);\n";
  }
  return result;
}

const std::string cuda_rt_decls =
    R"(
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind
declare i64 @get_thread_index();
declare i64 @get_block_index();
declare i32 @pos_start_impl(i32*);
declare i32 @group_buff_idx_impl();
declare i32 @pos_step_impl();
declare i8 @thread_warp_idx(i8);
declare i64* @init_shared_mem(i64*, i32);
declare i64* @init_shared_mem_nop(i64*, i32);
declare i64* @declare_dynamic_shared_memory();
declare void @write_back_nop(i64*, i64*, i32);
declare void @write_back_non_grouped_agg(i64*, i64*, i32);
declare void @init_group_by_buffer_gpu(i64*, i64*, i32, i32, i32, i1, i8);
declare i64* @get_group_value(i64*, i32, i64*, i32, i32, i32);
declare i64* @get_group_value_with_watchdog(i64*, i32, i64*, i32, i32, i32);
declare i32 @get_group_value_columnar_slot(i64*, i32, i64*, i32, i32);
declare i32 @get_group_value_columnar_slot_with_watchdog(i64*, i32, i64*, i32, i32);
declare i64* @get_group_value_fast(i64*, i64, i64, i64, i32);
declare i64* @get_group_value_fast_with_original_key(i64*, i64, i64, i64, i64, i32);
declare i32 @get_columnar_group_bin_offset(i64*, i64, i64, i64);
declare i64 @baseline_hash_join_idx_32(i8*, i8*, i64, i64);
declare i64 @baseline_hash_join_idx_64(i8*, i8*, i64, i64);
declare i64 @get_composite_key_index_32(i32*, i64, i32*, i64);
declare i64 @get_composite_key_index_64(i64*, i64, i64*, i64);
declare i64 @get_bucket_key_for_range_compressed(i8*, i64, double);
declare i64 @get_bucket_key_for_range_double(i8*, i64, double);
declare i32 @get_num_buckets_for_bounds(i8*, i32, double, double);
declare i64 @get_candidate_rows(i32*, i32, i8*, i32, double, double, i32, i64, i64*, i64, i64, i64);
declare i64 @agg_count_shared(i64*, i64);
declare i64 @agg_count_skip_val_shared(i64*, i64, i64);
declare i32 @agg_count_int32_shared(i32*, i32);
declare i32 @agg_count_int32_skip_val_shared(i32*, i32, i32);
declare i64 @agg_count_double_shared(i64*, double);
declare i64 @agg_count_double_skip_val_shared(i64*, double, double);
declare i32 @agg_count_float_shared(i32*, float);
declare i32 @agg_count_float_skip_val_shared(i32*, float, float);
declare i64 @agg_sum_shared(i64*, i64);
declare i64 @agg_sum_skip_val_shared(i64*, i64, i64);
declare i32 @agg_sum_int32_shared(i32*, i32);
declare i32 @agg_sum_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_sum_double_shared(i64*, double);
declare void @agg_sum_double_skip_val_shared(i64*, double, double);
declare void @agg_sum_float_shared(i32*, float);
declare void @agg_sum_float_skip_val_shared(i32*, float, float);
declare void @agg_max_shared(i64*, i64);
declare void @agg_max_skip_val_shared(i64*, i64, i64);
declare void @agg_max_int32_shared(i32*, i32);
declare void @agg_max_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_max_int16_shared(i16*, i16);
declare void @agg_max_int16_skip_val_shared(i16*, i16, i16);
declare void @agg_max_int8_shared(i8*, i8);
declare void @agg_max_int8_skip_val_shared(i8*, i8, i8);
declare void @agg_max_double_shared(i64*, double);
declare void @agg_max_double_skip_val_shared(i64*, double, double);
declare void @agg_max_float_shared(i32*, float);
declare void @agg_max_float_skip_val_shared(i32*, float, float);
declare void @agg_min_shared(i64*, i64);
declare void @agg_min_skip_val_shared(i64*, i64, i64);
declare void @agg_min_int32_shared(i32*, i32);
declare void @agg_min_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_min_int16_shared(i16*, i16);
declare void @agg_min_int16_skip_val_shared(i16*, i16, i16);
declare void @agg_min_int8_shared(i8*, i8);
declare void @agg_min_int8_skip_val_shared(i8*, i8, i8);
declare void @agg_min_double_shared(i64*, double);
declare void @agg_min_double_skip_val_shared(i64*, double, double);
declare void @agg_min_float_shared(i32*, float);
declare void @agg_min_float_skip_val_shared(i32*, float, float);
declare void @agg_id_shared(i64*, i64);
declare i8* @agg_id_varlen_shared(i8*, i64, i8*, i64);
declare void @agg_id_int32_shared(i32*, i32);
declare void @agg_id_int16_shared(i16*, i16);
declare void @agg_id_int8_shared(i8*, i8);
declare void @agg_id_double_shared(i64*, double);
declare void @agg_id_double_shared_slow(i64*, double*);
declare void @agg_id_float_shared(i32*, float);
declare i32 @checked_single_agg_id_shared(i64*, i64, i64);
declare i32 @checked_single_agg_id_double_shared(i64*, double, double);
declare i32 @checked_single_agg_id_double_shared_slow(i64*, double*, double);
declare i32 @checked_single_agg_id_float_shared(i32*, float, float);
declare i1 @slotEmptyKeyCAS(i64*, i64, i64);
declare i1 @slotEmptyKeyCAS_int32(i32*, i32, i32);
declare i1 @slotEmptyKeyCAS_int16(i16*, i16, i16);
declare i1 @slotEmptyKeyCAS_int8(i8*, i8, i8);
declare i64 @datetrunc_century(i64);
declare i64 @datetrunc_day(i64);
declare i64 @datetrunc_decade(i64);
declare i64 @datetrunc_hour(i64);
declare i64 @datetrunc_millennium(i64);
declare i64 @datetrunc_minute(i64);
declare i64 @datetrunc_month(i64);
declare i64 @datetrunc_quarter(i64);
declare i64 @datetrunc_quarterday(i64);
declare i64 @datetrunc_week_monday(i64);
declare i64 @datetrunc_week_sunday(i64);
declare i64 @datetrunc_week_saturday(i64);
declare i64 @datetrunc_year(i64);
declare i64 @extract_epoch(i64);
declare i64 @extract_dateepoch(i64);
declare i64 @extract_quarterday(i64);
declare i64 @extract_hour(i64);
declare i64 @extract_minute(i64);
declare i64 @extract_second(i64);
declare i64 @extract_millisecond(i64);
declare i64 @extract_microsecond(i64);
declare i64 @extract_nanosecond(i64);
declare i64 @extract_dow(i64);
declare i64 @extract_isodow(i64);
declare i64 @extract_day(i64);
declare i64 @extract_week_monday(i64);
declare i64 @extract_week_sunday(i64);
declare i64 @extract_week_saturday(i64);
declare i64 @extract_day_of_year(i64);
declare i64 @extract_month(i64);
declare i64 @extract_quarter(i64);
declare i64 @extract_year(i64);
declare i64 @DateTruncateHighPrecisionToDate(i64, i64);
declare i64 @DateTruncateHighPrecisionToDateNullable(i64, i64, i64);
declare i64 @DateDiff(i32, i64, i64);
declare i64 @DateDiffNullable(i32, i64, i64, i64);
declare i64 @DateDiffHighPrecision(i32, i64, i64, i32, i32);
declare i64 @DateDiffHighPrecisionNullable(i32, i64, i64, i32, i32, i64);
declare i64 @DateAdd(i32, i64, i64);
declare i64 @DateAddNullable(i32, i64, i64, i64);
declare i64 @DateAddHighPrecision(i32, i64, i64, i32);
declare i64 @DateAddHighPrecisionNullable(i32, i64, i64, i32, i64);
declare i64 @string_decode(i8*, i64);
declare i32 @array_size(i8*, i64, i32);
declare i32 @array_size_nullable(i8*, i64, i32, i32);
declare i32 @fast_fixlen_array_size(i8*, i32);
declare i1 @array_is_null(i8*, i64);
declare i8* @array_buff(i8*, i64);
declare i8* @fast_fixlen_array_buff(i8*, i64);
declare i8 @array_at_int8_t(i8*, i64, i32);
declare i16 @array_at_int16_t(i8*, i64, i32);
declare i32 @array_at_int32_t(i8*, i64, i32);
declare i64 @array_at_int64_t(i8*, i64, i32);
declare float @array_at_float(i8*, i64, i32);
declare double @array_at_double(i8*, i64, i32);
declare i8 @varlen_array_at_int8_t(i8*, i64, i32);
declare i16 @varlen_array_at_int16_t(i8*, i64, i32);
declare i32 @varlen_array_at_int32_t(i8*, i64, i32);
declare i64 @varlen_array_at_int64_t(i8*, i64, i32);
declare float @varlen_array_at_float(i8*, i64, i32);
declare double @varlen_array_at_double(i8*, i64, i32);
declare i8 @varlen_notnull_array_at_int8_t(i8*, i64, i32);
declare i16 @varlen_notnull_array_at_int16_t(i8*, i64, i32);
declare i32 @varlen_notnull_array_at_int32_t(i8*, i64, i32);
declare i64 @varlen_notnull_array_at_int64_t(i8*, i64, i32);
declare float @varlen_notnull_array_at_float(i8*, i64, i32);
declare double @varlen_notnull_array_at_double(i8*, i64, i32);
declare i8 @array_at_int8_t_checked(i8*, i64, i64, i8);
declare i16 @array_at_int16_t_checked(i8*, i64, i64, i16);
declare i32 @array_at_int32_t_checked(i8*, i64, i64, i32);
declare i64 @array_at_int64_t_checked(i8*, i64, i64, i64);
declare float @array_at_float_checked(i8*, i64, i64, float);
declare double @array_at_double_checked(i8*, i64, i64, double);
declare i32 @char_length(i8*, i32);
declare i32 @char_length_nullable(i8*, i32, i32);
declare i32 @char_length_encoded(i8*, i32);
declare i32 @char_length_encoded_nullable(i8*, i32, i32);
declare i32 @key_for_string_encoded(i32);
declare i1 @sample_ratio(double, i64);
declare double @width_bucket(double, double, double, double, i32);
declare double @width_bucket_reverse(double, double, double, double, i32);
declare double @width_bucket_nullable(double, double, double, double, i32, double);
declare double @width_bucket_reversed_nullable(double, double, double, double, i32, double);
declare double @width_bucket_no_oob_check(double, double, double);
declare double @width_bucket_reverse_no_oob_check(double, double, double);
declare double @width_bucket_expr(double, i1, double, double, i32);
declare double @width_bucket_expr_nullable(double, i1, double, double, i32, double);
declare double @width_bucket_expr_no_oob_check(double, i1, double, double, i32);
declare i1 @string_like(i8*, i32, i8*, i32, i8);
declare i1 @string_ilike(i8*, i32, i8*, i32, i8);
declare i8 @string_like_nullable(i8*, i32, i8*, i32, i8, i8);
declare i8 @string_ilike_nullable(i8*, i32, i8*, i32, i8, i8);
declare i1 @string_like_simple(i8*, i32, i8*, i32);
declare i1 @string_ilike_simple(i8*, i32, i8*, i32);
declare i8 @string_like_simple_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ilike_simple_nullable(i8*, i32, i8*, i32, i8);
declare i1 @string_lt(i8*, i32, i8*, i32);
declare i1 @string_le(i8*, i32, i8*, i32);
declare i1 @string_gt(i8*, i32, i8*, i32);
declare i1 @string_ge(i8*, i32, i8*, i32);
declare i1 @string_eq(i8*, i32, i8*, i32);
declare i1 @string_ne(i8*, i32, i8*, i32);
declare i8 @string_lt_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_le_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_gt_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ge_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_eq_nullable(i8*, i32, i8*, i32, i8);
declare i8 @string_ne_nullable(i8*, i32, i8*, i32, i8);
declare i1 @regexp_like(i8*, i32, i8*, i32, i8);
declare i8 @regexp_like_nullable(i8*, i32, i8*, i32, i8, i8);
declare void @linear_probabilistic_count(i8*, i32, i8*, i32);
declare void @agg_count_distinct_bitmap_gpu(i64*, i64, i64, i64, i64, i64, i64);
declare void @agg_count_distinct_bitmap_skip_val_gpu(i64*, i64, i64, i64, i64, i64, i64, i64);
declare void @agg_approximate_count_distinct_gpu(i64*, i64, i32, i64, i64);
declare void @record_error_code(i32, i32*);
declare i32 @get_error_code(i32*);
declare i1 @dynamic_watchdog();
declare i1 @check_interrupt();
declare void @force_sync();
declare void @sync_warp();
declare void @sync_warp_protected(i64, i64);
declare void @sync_threadblock();
declare i64* @get_bin_from_k_heap_int32_t(i64*, i32, i32, i32, i1, i1, i1, i32, i32);
declare i64* @get_bin_from_k_heap_int64_t(i64*, i32, i32, i32, i1, i1, i1, i64, i64);
declare i64* @get_bin_from_k_heap_float(i64*, i32, i32, i32, i1, i1, i1, float, float);
declare i64* @get_bin_from_k_heap_double(i64*, i32, i32, i32, i1, i1, i1, double, double);
)" + gen_array_any_all_sigs() +
    gen_translate_null_key_sigs();

}  // namespace

void CUDABackend::linkModuleWithLibdevice(const std::unique_ptr<llvm::Module>& ext,
                                          llvm::Module& llvm_module,
                                          llvm::PassManagerBuilder& pass_manager_builder,
                                          const GPUTarget& gpu_target,
                                          llvm::TargetMachine* nvptx_target_machine) {
#ifdef HAVE_CUDA
  auto timer = DEBUG_TIMER(__func__);

  if (!ext) {
    // raise error
    throw std::runtime_error(
        "libdevice library is not available but required by the UDF module");
  }

  // Saves functions \in module
  std::vector<llvm::Function*> roots;
  for (llvm::Function& fn : llvm_module) {
    if (!fn.isDeclaration())
      roots.emplace_back(&fn);
  }

  // Bind libdevice to the current module
  CodeGenerator::link_udf_module(
      ext, llvm_module, gpu_target.cgen_state, llvm::Linker::Flags::OverrideFromSrc);

  std::unordered_set<llvm::Function*> live_funcs =
      findAliveRuntimeFuncs(llvm_module, roots);

  std::vector<llvm::Function*> funcs_to_delete;
  for (llvm::Function& fn : llvm_module) {
    if (!live_funcs.count(&fn)) {
      // deleting the function were would invalidate the iterator
      funcs_to_delete.emplace_back(&fn);
    }
  }

  for (llvm::Function* f : funcs_to_delete) {
    f->eraseFromParent();
  }

  // activate nvvm-reflect-ftz flag on the module
#if LLVM_VERSION_MAJOR >= 11
  llvm::LLVMContext& ctx = llvm_module.getContext();
  llvm_module.setModuleFlag(llvm::Module::Override,
                            "nvvm-reflect-ftz",
                            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                llvm::Type::getInt32Ty(ctx), uint32_t(1))));
#else
  llvm_module.addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", uint32_t(1));
#endif
  for (llvm::Function& fn : llvm_module) {
    fn.addFnAttr("nvptx-f32ftz", "true");
  }

  // add nvvm reflect pass replacing any NVVM conditionals with constants
  nvptx_target_machine->adjustPassManager(pass_manager_builder);
  llvm::legacy::FunctionPassManager FPM(&llvm_module);
  pass_manager_builder.populateFunctionPassManager(FPM);

  // Run the NVVMReflectPass here rather than inside optimize_ir
  FPM.doInitialization();
  for (auto& F : llvm_module) {
    FPM.run(F);
  }
  FPM.doFinalization();
#endif
}

std::shared_ptr<CudaCompilationContext> CUDABackend::generateNativeGPUCode(
    const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const bool is_gpu_smem_used,
    const CompilationOptions& co,
    const GPUTarget& gpu_target,
    llvm::TargetMachine* nvptx_target_machine) {
#ifdef HAVE_CUDA
  auto timer = DEBUG_TIMER(__func__);
  auto llvm_module = func->getParent();
  /*
    `func` is one of the following generated functions:
    - `call_table_function(i8** %input_col_buffers, i64*
      %input_row_count, i64** %output_buffers, i64* %output_row_count)`
      that wraps the user-defined table function.
    - `multifrag_query`
    - `multifrag_query_hoisted_literals`
    - ...

    `wrapper_func` is table_func_kernel(i32*, i8**, i64*, i64**,
    i64*) that wraps `call_table_function`.

    `llvm_module` is from `build/QueryEngine/RuntimeFunctions.bc` and it
    contains `func` and `wrapper_func`.  `module` should also contain
    the definitions of user-defined table functions.

    `live_funcs` contains table_func_kernel and call_table_function

    `gpu_target.cgen_state->module_` appears to be the same as `llvm_module`
   */
  CHECK(gpu_target.cgen_state->module_ == llvm_module);
  llvm_module->setDataLayout(compiler::get_gpu_data_layout());
  llvm_module->setTargetTriple(compiler::get_gpu_target_triple_string());
  llvm::PassManagerBuilder pass_manager_builder = llvm::PassManagerBuilder();

  pass_manager_builder.OptLevel = 0;
  llvm::legacy::PassManager module_pass_manager;
  pass_manager_builder.populateModulePassManager(module_pass_manager);

  bool requires_libdevice = check_module_requires_libdevice(llvm_module);

  if (requires_libdevice) {
    linkModuleWithLibdevice(exts.at(ExtModuleKinds::rt_libdevice_module),
                            *llvm_module,
                            pass_manager_builder,
                            gpu_target,
                            nvptx_target_machine);
  }

  // run optimizations
  compiler::optimize_ir(
      func, llvm_module, module_pass_manager, live_funcs, is_gpu_smem_used, co);
  compiler::legalize_nvvm_ir(func);

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);

  llvm::LLVMContext& ctx = llvm_module->getContext();
  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode* md = llvm_module->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata* md_vals[] = {llvm::ConstantAsMetadata::get(wrapper_func),
                               llvm::MDString::get(ctx, "kernel"),
                               llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                   llvm::Type::getInt32Ty(ctx), 1))};

  // Append metadata to nvvm.annotations
  md->addOperand(llvm::MDNode::get(ctx, md_vals));

  std::unordered_set<llvm::Function*> roots{wrapper_func, func};
  if (gpu_target.row_func_not_inlined) {
    clear_function_attributes(gpu_target.cgen_state->row_func_);
    roots.insert(gpu_target.cgen_state->row_func_);
    if (gpu_target.cgen_state->filter_func_) {
      roots.insert(gpu_target.cgen_state->filter_func_);
    }
  }

  // prevent helper functions from being removed
  for (auto f : gpu_target.cgen_state->helper_functions_) {
    roots.insert(f);
  }

  if (requires_libdevice) {
    for (llvm::Function& F : *llvm_module) {
      // Some libdevice functions calls another functions that starts with "__internal_"
      // prefix.
      // __internal_trig_reduction_slowpathd
      // __internal_accurate_pow
      // __internal_lgamma_pos
      // Those functions have a "noinline" attribute which prevents the optimizer from
      // inlining them into the body of @query_func
      if (F.hasName() && F.getName().startswith("__internal") && !F.isDeclaration()) {
        roots.insert(&F);
      }
      legalize_nvvm_ir(&F);
    }
  }

  // Prevent the udf function(s) from being removed the way the runtime functions are
  std::unordered_set<std::string> udf_declarations;

  if (exts.find(ExtModuleKinds::udf_gpu_module) != exts.end()) {
    for (auto& f : exts.at(ExtModuleKinds::udf_gpu_module)->getFunctionList()) {
      llvm::Function* udf_function = llvm_module->getFunction(f.getName());

      if (udf_function) {
        legalize_nvvm_ir(udf_function);
        roots.insert(udf_function);

        // If we have a udf that declares a external function
        // note it so we can avoid duplicate declarations
        if (f.isDeclaration()) {
          udf_declarations.insert(f.getName().str());
        }
      }
    }
  }

  if (exts.find(ExtModuleKinds::rt_udf_gpu_module) != exts.end()) {
    for (auto& f : exts.at(ExtModuleKinds::rt_udf_gpu_module)->getFunctionList()) {
      llvm::Function* udf_function = llvm_module->getFunction(f.getName());
      if (udf_function) {
        legalize_nvvm_ir(udf_function);
        roots.insert(udf_function);

        // If we have a udf that declares a external function
        // note it so we can avoid duplicate declarations
        if (f.isDeclaration()) {
          udf_declarations.insert(f.getName().str());
        }
      }
    }
  }

  std::vector<llvm::Function*> rt_funcs;
  for (auto& Fn : *llvm_module) {
    if (roots.count(&Fn)) {
      continue;
    }
    rt_funcs.push_back(&Fn);
  }
  for (auto& pFn : rt_funcs) {
    pFn->removeFromParent();
  }

  if (requires_libdevice) {
    add_intrinsics_to_module(llvm_module);
  }

  llvm_module->print(os, nullptr);
  os.flush();

  for (auto& pFn : rt_funcs) {
    llvm_module->getFunctionList().push_back(pFn);
  }
  llvm_module->eraseNamedMetadata(md);

  auto cuda_llir = ss.str() + cuda_rt_decls + extension_function_decls(udf_declarations);
  std::string ptx;
  try {
    ptx = generatePTX(cuda_llir, nvptx_target_machine, gpu_target.cgen_state->context_);
  } catch (ParseIRError& e) {
    LOG(WARNING) << "Failed to generate PTX: " << e.what()
                 << ". Switching to CPU execution target.";
    throw QueryMustRunOnCpu();
  }
  LOG(PTX) << "PTX for the GPU:\n" << ptx << "\nEnd of PTX";

  const auto cuda_mgr =
      dynamic_cast<const CudaMgr_Namespace::CudaMgr*>(gpu_target.gpu_mgr);
  auto cubin_result = ptx_to_cubin(ptx, gpu_target.block_size, cuda_mgr);
  auto& option_keys = cubin_result.option_keys;
  auto& option_values = cubin_result.option_values;
  auto cubin = cubin_result.cubin;
  auto link_state = cubin_result.link_state;
  const auto num_options = option_keys.size();

  auto func_name = wrapper_func->getName().str();
  auto gpu_compilation_context = std::make_shared<CudaCompilationContext>();
  for (int device_id = 0; device_id < gpu_target.gpu_mgr->getDeviceCount(); ++device_id) {
    gpu_compilation_context->addDeviceCode(
        std::make_unique<CudaDeviceCompilationContext>(cubin,
                                                       func_name,
                                                       device_id,
                                                       cuda_mgr,
                                                       num_options,
                                                       &option_keys[0],
                                                       &option_values[0]));
  }

  checkCudaErrors(cuLinkDestroy(link_state));
  return gpu_compilation_context;
#else
  return {};
#endif
}

std::shared_ptr<CompilationContext> L0Backend::generateNativeCode(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  return generateNativeGPUCode(func, wrapper_func, live_funcs, co, gpu_target_);
}

std::shared_ptr<L0CompilationContext> L0Backend::generateNativeGPUCode(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co,
    const GPUTarget& gpu_target) {
#ifdef HAVE_L0
  auto module = func->getParent();

  CHECK(module);
  CHECK(wrapper_func);

  for (auto& Fn : *module) {
    Fn.setCallingConv(llvm::CallingConv::SPIR_FUNC);
  }
  wrapper_func->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

  for (auto& Fn : *module) {
    for (auto I = llvm::inst_begin(Fn), E = llvm::inst_end(Fn); I != E; ++I) {
      if (auto* CI = llvm::dyn_cast<llvm::CallInst>(&*I)) {
        CI->setCallingConv(llvm::CallingConv::SPIR_FUNC);
      }
    }
  }

  auto pass_manager_builder = llvm::PassManagerBuilder();
  llvm::legacy::PassManager PM;
  pass_manager_builder.populateModulePassManager(PM);
  compiler::optimize_ir(func, module, PM, live_funcs, false /*smem_used*/, co);

  module->setTargetTriple("spir64-unknown-unknown");

  llvm::LLVMContext& ctx = module->getContext();
  // set metadata -- pretend we're opencl (see
  // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst#spir-v-instructions-mapped-to-llvm-metadata)
  llvm::Metadata* spirv_src_ops[] = {
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 3 /*OpenCL_C*/)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx),
                                                           102000 /*OpenCL ver 1.2*/))};
  llvm::NamedMDNode* spirv_src = module->getOrInsertNamedMetadata("spirv.Source");
  spirv_src->addOperand(llvm::MDNode::get(ctx, spirv_src_ops));

  SPIRV::TranslatorOpts opts;
  opts.enableAllExtensions();
  opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL12);
  opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);

  std::ostringstream ss;
  std::string err;
  auto success = writeSpirv(module, opts, ss, err);
  CHECK(success) << "Spirv translation failed with error: " << err << "\n";

  const auto func_name = wrapper_func->getName().str();
  L0BinResult bin_result;
  const auto l0_mgr = dynamic_cast<const l0::L0Manager*>(gpu_target.gpu_mgr);
  try {
    bin_result = spv_to_bin(ss.str(), func_name, 1 /*todo block size*/, l0_mgr);
  } catch (l0::L0Exception& e) {
    llvm::errs() << e.what() << "\n";
    return {};
  }

  auto compilation_ctx = std::make_shared<L0CompilationContext>();
  auto device_compilation_ctx = std::make_unique<L0DeviceCompilationContext>(
      bin_result.device, bin_result.kernel, bin_result.module, l0_mgr, 0, 1);
  compilation_ctx->addDeviceCode(move(device_compilation_ctx));
  return compilation_ctx;
#else
  return {};
#endif  // HAVE_L0
}

std::shared_ptr<Backend> getBackend(
    ExecutorDeviceType dt,
    const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
    bool is_gpu_smem_used_,
    GPUTarget& gpu_target) {
  switch (dt) {
    case ExecutorDeviceType::CPU:
      return std::make_shared<CPUBackend>();
    case ExecutorDeviceType::GPU:
      if (gpu_target.gpu_mgr->getPlatform() == GpuMgrPlatform::CUDA)
        return std::make_shared<CUDABackend>(exts, is_gpu_smem_used_, gpu_target);
      if (gpu_target.gpu_mgr->getPlatform() == GpuMgrPlatform::L0)
        return std::make_shared<L0Backend>(gpu_target);
    default:
      CHECK(false);
      return {};
  };
}
}  // namespace compiler
