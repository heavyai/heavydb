/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "Execute.h"
#include "ExtensionFunctionsWhitelist.h"
#include "QueryTemplateGenerator.h"

#include "Shared/mapdpath.h"

#if LLVM_VERSION_MAJOR >= 4
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#else
#include <llvm/Bitcode/ReaderWriter.h>
#endif
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/InstIterator.h>
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/IPO.h>
#if LLVM_VERSION_MAJOR >= 4
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#endif
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

namespace {

void eliminateDeadSelfRecursiveFuncs(llvm::Module& M, std::unordered_set<llvm::Function*>& live_funcs) {
  std::vector<llvm::Function*> dead_funcs;
  for (auto& F : M) {
    bool bAlive = false;
    if (live_funcs.count(&F))
      continue;
    for (auto U : F.users()) {
      auto* C = llvm::dyn_cast<const llvm::CallInst>(U);
      if (!C || C->getParent()->getParent() != &F) {
        bAlive = true;
        break;
      }
    }
    if (!bAlive)
      dead_funcs.push_back(&F);
  }
  for (auto pFn : dead_funcs) {
    pFn->eraseFromParent();
  }
}

void verify_function_ir(const llvm::Function* func) {
  std::stringstream err_ss;
  llvm::raw_os_ostream err_os(err_ss);
  if (llvm::verifyFunction(*func, &err_os)) {
    func->dump();
    LOG(FATAL) << err_ss.str();
  }
}

void optimizeIR(llvm::Function* query_func,
                llvm::Module* module,
                std::unordered_set<llvm::Function*>& live_funcs,
                const CompilationOptions& co,
                const std::string& debug_dir,
                const std::string& debug_file) {
  llvm::legacy::PassManager pass_manager;
#if LLVM_VERSION_MAJOR < 4
  pass_manager.add(llvm::createAlwaysInlinerPass());
#else
  pass_manager.add(llvm::createAlwaysInlinerLegacyPass());
#endif
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.add(llvm::createInstructionSimplifierPass());
  pass_manager.add(llvm::createInstructionCombiningPass());
  pass_manager.add(llvm::createGlobalOptimizerPass());
// FIXME(miyu): need investigate how 3.7+ dump debug IR.
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  if (!debug_dir.empty()) {
    CHECK(!debug_file.empty());
    pass_manager.add(llvm::createDebugIRPass(false, false, debug_dir, debug_file));
  }
#endif
  if (co.hoist_literals_) {
    pass_manager.add(llvm::createLICMPass());
  }
  if (co.opt_level_ == ExecutorOptLevel::LoopStrengthReduction) {
    pass_manager.add(llvm::createLoopStrengthReducePass());
  }
  pass_manager.run(*module);

  eliminateDeadSelfRecursiveFuncs(*module, live_funcs);

  // optimizations might add attributes to the function
  // and NVPTX doesn't understand all of them; play it
  // safe and clear all attributes
  llvm::AttributeSet no_attributes;
  query_func->setAttributes(no_attributes);
  verify_function_ir(query_func);
}

template <class T>
std::string serialize_llvm_object(const T* llvm_obj) {
  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  llvm_obj->print(os);
  os.flush();
  return ss.str();
}

}  // namespace

std::vector<std::pair<void*, void*>> Executor::getCodeFromCache(
    const CodeCacheKey& key,
    const std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>& cache) {
  auto it = cache.find(key);
  if (it != cache.end()) {
    delete cgen_state_->module_;
    cgen_state_->module_ = it->second.second;
    std::vector<std::pair<void*, void*>> native_functions;
    for (auto& native_code : it->second.first) {
      GpuCompilationContext* gpu_context = std::get<2>(native_code).get();
      native_functions.push_back(
          std::make_pair(std::get<0>(native_code), gpu_context ? gpu_context->module() : nullptr));
    }
    return native_functions;
  }
  return {};
}

void Executor::addCodeToCache(
    const CodeCacheKey& key,
    const std::vector<std::tuple<void*, llvm::ExecutionEngine*, GpuCompilationContext*>>& native_code,
    llvm::Module* module,
    std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>& cache) {
  CHECK(!native_code.empty());
  CodeCacheVal cache_val;
  for (const auto& native_func : native_code) {
    cache_val.emplace_back(std::get<0>(native_func),
                           std::unique_ptr<llvm::ExecutionEngine>(std::get<1>(native_func)),
                           std::unique_ptr<GpuCompilationContext>(std::get<2>(native_func)));
  }
  auto it_ok = cache.insert(std::make_pair(key, std::make_pair(std::move(cache_val), module)));
  CHECK(it_ok.second);
}

std::vector<std::pair<void*, void*>> Executor::optimizeAndCodegenCPU(llvm::Function* query_func,
                                                                     llvm::Function* multifrag_query_func,
                                                                     std::unordered_set<llvm::Function*>& live_funcs,
                                                                     llvm::Module* module,
                                                                     const CompilationOptions& co) {
  CodeCacheKey key{serialize_llvm_object(query_func), serialize_llvm_object(cgen_state_->row_func_)};
  for (const auto helper : cgen_state_->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  auto cached_code = getCodeFromCache(key, cpu_code_cache_);
  if (!cached_code.empty()) {
    return cached_code;
  }

  // run optimizations
  optimizeIR(query_func, module, live_funcs, co, debug_dir_, debug_file_);

  llvm::ExecutionEngine* execution_engine{nullptr};

  auto init_err = llvm::InitializeNativeTarget();
  CHECK(!init_err);

  llvm::InitializeAllTargetMCs();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  std::string err_str;
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  llvm::EngineBuilder eb(module);
  eb.setUseMCJIT(true);
#else
  std::unique_ptr<llvm::Module> owner(module);
  llvm::EngineBuilder eb(std::move(owner));
#endif
  eb.setErrorStr(&err_str);
  eb.setEngineKind(llvm::EngineKind::JIT);
  llvm::TargetOptions to;
  to.EnableFastISel = true;
  eb.setTargetOptions(to);
  execution_engine = eb.create();
  CHECK(execution_engine);

  execution_engine->finalizeObject();
  auto native_code = execution_engine->getPointerToFunction(multifrag_query_func);

  CHECK(native_code);
  addCodeToCache(key, {{std::make_tuple(native_code, execution_engine, nullptr)}}, module, cpu_code_cache_);

  return {std::make_pair(native_code, nullptr)};
}

namespace {

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
    for (const std::string elem_type : {"int8_t", "int16_t", "int32_t", "int64_t", "float", "double"}) {
      for (const std::string needle_type : {"int8_t", "int16_t", "int32_t", "int64_t", "float", "double"}) {
        for (const std::string op_name : {"eq", "ne", "lt", "le", "gt", "ge"}) {
          result += ("declare i1 @array_" + any_or_all + "_" + op_name + "_" + elem_type + "_" + needle_type +
                     "(i8*, i64, " + cpp_to_llvm_name(needle_type) + ", " + cpp_to_llvm_name(elem_type) + ");\n");
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
    result += "declare i64 @translate_null_key_" + key_type + "(" + key_llvm_type + ", " + key_llvm_type + ", " +
              key_llvm_type + ");\n";
  }
  return result;
}

const std::string cuda_rt_decls =
    R"(
declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
declare i32 @pos_start_impl(i32*);
declare i32 @group_buff_idx_impl();
declare i32 @pos_step_impl();
declare i8 @thread_warp_idx(i8);
declare i64* @init_shared_mem(i64*, i32);
declare i64* @init_shared_mem_nop(i64*, i32);
declare void @write_back(i64*, i64*, i32);
declare void @write_back_nop(i64*, i64*, i32);
declare void @init_group_by_buffer_gpu(i64*, i64*, i32, i32, i32, i1, i8);
declare i64* @get_group_value(i64*, i32, i64*, i32, i32, i32, i64*);
declare i64* @get_group_value_with_watchdog(i64*, i32, i64*, i32, i32, i32, i64*);
declare i64* @get_group_value_fast(i64*, i64, i64, i64, i32);
declare i64* @get_group_value_fast_with_original_key(i64*, i64, i64, i64, i64, i32);
declare i32 @get_columnar_group_bin_offset(i64*, i64, i64, i64);
declare i64* @get_group_value_one_key(i64*, i32, i64*, i32, i64, i64, i32, i64*);
declare i64* @get_group_value_one_key_with_watchdog(i64*, i32, i64*, i32, i64, i64, i32, i64*);
declare i64 @baseline_hash_join_idx_32(i8*, i8*, i64, i64);
declare i64 @baseline_hash_join_idx_64(i8*, i8*, i64, i64);
declare i64 @get_composite_key_index_32(i32*, i64, i32*, i64);
declare i64 @get_composite_key_index_64(i64*, i64, i64*, i64);
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
declare void @agg_max_double_shared(i64*, double);
declare void @agg_max_double_skip_val_shared(i64*, double, double);
declare void @agg_max_float_shared(i32*, float);
declare void @agg_max_float_skip_val_shared(i32*, float, float);
declare void @agg_min_shared(i64*, i64);
declare void @agg_min_skip_val_shared(i64*, i64, i64);
declare void @agg_min_int32_shared(i32*, i32);
declare void @agg_min_int32_skip_val_shared(i32*, i32, i32);
declare void @agg_min_double_shared(i64*, double);
declare void @agg_min_double_skip_val_shared(i64*, double, double);
declare void @agg_min_float_shared(i32*, float);
declare void @agg_min_float_skip_val_shared(i32*, float, float);
declare void @agg_id_shared(i64*, i64);
declare void @agg_id_int32_shared(i32*, i32);
declare void @agg_id_double_shared(i64*, double);
declare void @agg_id_double_shared_slow(i64*, double*);
declare void @agg_id_float_shared(i32*, float);
declare i64 @ExtractFromTime(i32, i64);
declare i64 @ExtractFromTimeNullable(i32, i64, i64);
declare i64 @DateTruncate(i32, i64);
declare i64 @DateTruncateNullable(i32, i64, i64);
declare i64 @DateDiff(i32, i64, i64);
declare i64 @DateDiffNullable(i32, i64, i64, i64);
declare i64 @DateAdd(i32, i64, i64);
declare i64 @DateAddNullable(i32, i64, i64, i64);
declare i64 @string_decode(i8*, i64);
declare i32 @array_size(i8*, i64, i32);
declare i1 @array_is_null(i8*, i64);
declare i8* @array_buff(i8*, i64);
declare i8 @array_at_int8_t(i8*, i64, i32);
declare i16 @array_at_int16_t(i8*, i64, i32);
declare i32 @array_at_int32_t(i8*, i64, i32);
declare i64 @array_at_int64_t(i8*, i64, i32);
declare float @array_at_float(i8*, i64, i32);
declare double @array_at_double(i8*, i64, i32);
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
declare i32 @record_error_code(i32, i32*);
declare i1 @dynamic_watchdog();
declare void @force_sync();
declare i64* @get_bin_from_k_heap_int32_t(i64*, i32, i32, i32, i1, i1, i1, i32, i32);
declare i64* @get_bin_from_k_heap_int64_t(i64*, i32, i32, i32, i1, i1, i1, i64, i64);
declare i64* @get_bin_from_k_heap_float(i64*, i32, i32, i32, i1, i1, i1, float, float);
declare i64* @get_bin_from_k_heap_double(i64*, i32, i32, i32, i1, i1, i1, double, double);
)" + gen_array_any_all_sigs() +
    gen_translate_null_key_sigs();

#ifdef HAVE_CUDA
std::string extension_function_decls() {
  const auto decls = ExtensionFunctionsWhitelist::getLLVMDeclarations();
  return boost::algorithm::join(decls, "\n");
}

void legalize_nvvm_ir(llvm::Function* query_func) {
  std::vector<llvm::Instruction*> unsupported_intrinsics;
  for (auto& BB : *query_func) {
    for (llvm::Instruction& I : BB) {
      if (const llvm::IntrinsicInst* II = llvm::dyn_cast<llvm::IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == llvm::Intrinsic::stacksave ||
            II->getIntrinsicID() == llvm::Intrinsic::stackrestore) {
          unsupported_intrinsics.push_back(&I);
        }
      }
    }
  }

  for (auto& II : unsupported_intrinsics) {
    II->eraseFromParent();
  }
}
#endif  // HAVE_CUDA

}  // namespace

std::vector<std::pair<void*, void*>> Executor::optimizeAndCodegenGPU(llvm::Function* query_func,
                                                                     llvm::Function* multifrag_query_func,
                                                                     std::unordered_set<llvm::Function*>& live_funcs,
                                                                     llvm::Module* module,
                                                                     const bool no_inline,
                                                                     const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                                                     const CompilationOptions& co) {
#ifdef HAVE_CUDA
  CHECK(cuda_mgr);
  CodeCacheKey key{serialize_llvm_object(query_func), serialize_llvm_object(cgen_state_->row_func_)};
  for (const auto helper : cgen_state_->helper_functions_) {
    key.push_back(serialize_llvm_object(helper));
  }
  auto cached_code = getCodeFromCache(key, gpu_code_cache_);
  if (!cached_code.empty()) {
    return cached_code;
  }

  auto get_group_value_func = module->getFunction("get_group_value_one_key");
  CHECK(get_group_value_func);
  get_group_value_func->setAttributes(llvm::AttributeSet{});

  bool row_func_not_inlined = false;
  if (no_inline) {
    for (auto it = llvm::inst_begin(cgen_state_->row_func_), e = llvm::inst_end(cgen_state_->row_func_); it != e;
         ++it) {
      if (llvm::isa<llvm::CallInst>(*it)) {
        auto& get_gv_call = llvm::cast<llvm::CallInst>(*it);
        if (get_gv_call.getCalledFunction()->getName() == "get_group_value" ||
            get_gv_call.getCalledFunction()->getName() == "get_group_value_with_watchdog" ||
            get_gv_call.getCalledFunction()->getName() == "get_matching_group_value_perfect_hash" ||
            get_gv_call.getCalledFunction()->getName() == "string_decode" ||
            get_gv_call.getCalledFunction()->getName() == "array_size" ||
            get_gv_call.getCalledFunction()->getName() == "linear_probabilistic_count") {
          llvm::AttributeSet no_inline_attrs;
          no_inline_attrs = no_inline_attrs.addAttribute(cgen_state_->context_, 0, llvm::Attribute::NoInline);
          cgen_state_->row_func_->setAttributes(no_inline_attrs);
          row_func_not_inlined = true;
          break;
        }
      }
    }
  }

  module->setDataLayout(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
  module->setTargetTriple("nvptx64-nvidia-cuda");

  // run optimizations
  optimizeIR(query_func, module, live_funcs, co, "", "");

  legalize_nvvm_ir(query_func);

  std::stringstream ss;
  llvm::raw_os_ostream os(ss);

  llvm::LLVMContext& ctx = module->getContext();
  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode* md = module->getOrInsertNamedMetadata("nvvm.annotations");

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  llvm::Value* md_vals[] = {
      multifrag_query_func, llvm::MDString::get(ctx, "kernel"), llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1)};
#else
  llvm::Metadata* md_vals[] = {llvm::ConstantAsMetadata::get(multifrag_query_func),
                               llvm::MDString::get(ctx, "kernel"),
                               llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
#endif
  // Append metadata to nvvm.annotations
  md->addOperand(llvm::MDNode::get(ctx, md_vals));

  std::unordered_set<llvm::Function*> roots{multifrag_query_func, query_func};
  if (row_func_not_inlined) {
    llvm::AttributeSet no_attributes;
    cgen_state_->row_func_->setAttributes(no_attributes);
    roots.insert(cgen_state_->row_func_);
  }

  std::vector<llvm::Function*> rt_funcs;
  for (auto& Fn : *module) {
    if (roots.count(&Fn))
      continue;
    rt_funcs.push_back(&Fn);
  }
  for (auto& pFn : rt_funcs)
    pFn->removeFromParent();
  module->print(os, nullptr);
  os.flush();
  for (auto& pFn : rt_funcs) {
    module->getFunctionList().push_back(pFn);
  }
  module->eraseNamedMetadata(md);

  auto cuda_llir = cuda_rt_decls + extension_function_decls() + ss.str();

  std::vector<std::pair<void*, void*>> native_functions;
  std::vector<std::tuple<void*, llvm::ExecutionEngine*, GpuCompilationContext*>> cached_functions;

  const auto ptx = generatePTX(cuda_llir);

  auto cubin_result = ptx_to_cubin(ptx, blockSize(), cuda_mgr);
  auto& option_keys = cubin_result.option_keys;
  auto& option_values = cubin_result.option_values;
  auto cubin = cubin_result.cubin;
  auto link_state = cubin_result.link_state;
  const auto num_options = option_keys.size();

  auto func_name = multifrag_query_func->getName().str();
  for (int device_id = 0; device_id < cuda_mgr->getDeviceCount(); ++device_id) {
    auto gpu_context = new GpuCompilationContext(
        cubin, func_name, device_id, cuda_mgr, num_options, &option_keys[0], &option_values[0]);
    auto native_code = gpu_context->kernel();
    auto native_module = gpu_context->module();
    CHECK(native_code);
    CHECK(native_module);
    native_functions.push_back(std::make_pair(native_code, native_module));
    cached_functions.emplace_back(native_code, nullptr, gpu_context);
  }
  addCodeToCache(key, cached_functions, module, gpu_code_cache_);

  checkCudaErrors(cuLinkDestroy(link_state));

  return native_functions;
#else
  return {};
#endif
}

std::string Executor::generatePTX(const std::string& cuda_llir) const {
  initializeNVPTXBackend();
  auto mem_buff = llvm::MemoryBuffer::getMemBuffer(cuda_llir, "", false);

  llvm::SMDiagnostic err;

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  auto module = llvm::ParseIR(mem_buff, err, cgen_state_->context_);
#else
  auto module = llvm::parseIR(mem_buff->getMemBufferRef(), err, cgen_state_->context_);
#endif
  if (!module) {
    LOG(FATAL) << err.getMessage().str();
  }

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  std::stringstream ss;
  llvm::raw_os_ostream raw_os(ss);
  llvm::formatted_raw_ostream formatted_os(raw_os);
#else
  llvm::SmallString<256> code_str;
  llvm::raw_svector_ostream formatted_os(code_str);
#endif
  CHECK(nvptx_target_machine_);
  {
    llvm::legacy::PassManager ptxgen_pm;
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
    ptxgen_pm.add(new llvm::DataLayoutPass(module));
#else
    module->setDataLayout(nvptx_target_machine_->createDataLayout());
#endif

    nvptx_target_machine_->addPassesToEmitFile(ptxgen_pm, formatted_os, llvm::TargetMachine::CGFT_AssemblyFile);
    ptxgen_pm.run(*module);
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
    formatted_os.flush();
#endif
  }

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  return ss.str();
#else
  return code_str.str();
#endif
}

void Executor::initializeNVPTXBackend() const {
  if (nvptx_target_machine_) {
    return;
  }
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  std::string err;
  auto target = llvm::TargetRegistry::lookupTarget("nvptx64", err);
  if (!target) {
    LOG(FATAL) << err;
  }
  nvptx_target_machine_.reset(
      target->createTargetMachine("nvptx64-nvidia-cuda", "sm_30", "", llvm::TargetOptions(), llvm::Reloc::Static));
}

namespace {

llvm::Module* read_template_module(llvm::LLVMContext& context) {
  llvm::SMDiagnostic err;

  auto buffer_or_error = llvm::MemoryBuffer::getFile(mapd_root_abs_path() + "/QueryEngine/RuntimeFunctions.bc");
  CHECK(!buffer_or_error.getError());
  llvm::MemoryBuffer* buffer = buffer_or_error.get().get();
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 5
  auto module = llvm::parseBitcodeFile(buffer, context).get();
#else
  auto owner = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
#if LLVM_VERSION_MAJOR < 4
  CHECK(!owner.getError());
#else
  CHECK(!owner.takeError());
#endif
  auto module = owner.get().release();
#endif
  CHECK(module);

  return module;
}

void bind_pos_placeholders(const std::string& pos_fn_name,
                           const bool use_resume_param,
                           llvm::Function* query_func,
                           llvm::Module* module) {
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& pos_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(pos_call.getCalledFunction()->getName()) == pos_fn_name) {
      if (use_resume_param) {
        const auto error_code_arg = get_arg_by_name(query_func, "error_code");
        llvm::ReplaceInstWithInst(&pos_call,
                                  llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl"), error_code_arg));
      } else {
        llvm::ReplaceInstWithInst(&pos_call, llvm::CallInst::Create(module->getFunction(pos_fn_name + "_impl")));
      }
      break;
    }
  }
}

std::vector<llvm::Value*> generate_column_heads_load(const int num_columns,
                                                     llvm::Function* query_func,
                                                     llvm::LLVMContext& context) {
  auto max_col_local_id = num_columns - 1;
  auto& fetch_bb = query_func->front();
  llvm::IRBuilder<> fetch_ir_builder(&fetch_bb);
  fetch_ir_builder.SetInsertPoint(&*fetch_bb.begin());
  auto& in_arg_list = query_func->getArgumentList();
  CHECK_GE(in_arg_list.size(), size_t(4));
  auto& byte_stream_arg = in_arg_list.front();
  std::vector<llvm::Value*> col_heads;
  for (int col_id = 0; col_id <= max_col_local_id; ++col_id) {
    col_heads.emplace_back(fetch_ir_builder.CreateLoad(
        fetch_ir_builder.CreateGEP(&byte_stream_arg, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), col_id))));
  }
  return col_heads;
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
    arg_it->setName("small_group_by_buff");
    ++arg_it;
    arg_it->setName("crt_match");
    ++arg_it;
    arg_it->setName("total_matched");
    ++arg_it;
    arg_it->setName("old_total_matched");
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

std::pair<llvm::Function*, std::vector<llvm::Value*>> create_row_function(const size_t in_col_count,
                                                                          const size_t agg_col_count,
                                                                          const bool hoist_literals,
                                                                          llvm::Function* query_func,
                                                                          llvm::Module* module,
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
    // small group by buffer
    row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));
    // current match count
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // total match count passed from the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
    // old total match count returned to the caller
    row_process_arg_types.push_back(llvm::Type::getInt32PtrTy(context));
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

  // Generate the function signature and column head fetches s.t.
  // double indirection isn't needed in the inner loop
  auto col_heads = generate_column_heads_load(in_col_count, query_func, context);
  CHECK_EQ(in_col_count, col_heads.size());

  // column buffer arguments
  for (size_t i = 0; i < in_col_count; ++i) {
    row_process_arg_types.emplace_back(llvm::Type::getInt8PtrTy(context));
  }

  // join hash table argument
  row_process_arg_types.push_back(llvm::Type::getInt64PtrTy(context));

  // generate the function
  auto ft = llvm::FunctionType::get(get_int_type(32, context), row_process_arg_types, false);

  auto row_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "row_func", module);

  // set the row function argument names; for debugging purposes only
  set_row_func_argnames(row_func, in_col_count, agg_col_count, hoist_literals);

  return std::make_pair(row_func, col_heads);
}

void bind_query(llvm::Function* query_func,
                const std::string& query_fname,
                llvm::Function* multifrag_query_func,
                llvm::Module* module) {
  std::vector<llvm::CallInst*> query_stubs;
  for (auto it = llvm::inst_begin(multifrag_query_func), e = llvm::inst_end(multifrag_query_func); it != e; ++it) {
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

std::vector<std::string> get_agg_fnames(const std::vector<Analyzer::Expr*>& target_exprs, const bool is_group_by) {
  std::vector<std::string> result;
  for (size_t target_idx = 0, agg_col_idx = 0; target_idx < target_exprs.size(); ++target_idx, ++agg_col_idx) {
    const auto target_expr = target_exprs[target_idx];
    CHECK(target_expr);
    const auto target_type_info = target_expr->get_type_info();
    const auto target_type = target_type_info.get_type();
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      result.push_back((target_type == kFLOAT || target_type == kDOUBLE) ? "agg_id_double" : "agg_id");
      if (target_type_info.is_string() && target_type_info.get_compression() == kENCODING_NONE) {
        result.push_back("agg_id");
      }
      continue;
    }
    const auto agg_type = agg_expr->get_aggtype();
    const auto& agg_type_info = agg_type != kCOUNT ? agg_expr->get_arg()->get_type_info() : target_type_info;
    switch (agg_type) {
      case kAVG: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() && !agg_type_info.is_fp()) {
          throw std::runtime_error("AVG is only valid on integer and floating point");
        }
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_sum" : "agg_sum_double");
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_count" : "agg_count_double");
        break;
      }
      case kMIN: {
        if (agg_type_info.is_string() || agg_type_info.is_array()) {
          throw std::runtime_error("MIN on strings or arrays not supported yet");
        }
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_min" : "agg_min_double");
        break;
      }
      case kMAX: {
        if (agg_type_info.is_string() || agg_type_info.is_array()) {
          throw std::runtime_error("MAX on strings or arrays not supported yet");
        }
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_max" : "agg_max_double");
        break;
      }
      case kSUM: {
        if (!agg_type_info.is_integer() && !agg_type_info.is_decimal() && !agg_type_info.is_fp()) {
          throw std::runtime_error("SUM is only valid on integer and floating point");
        }
        result.push_back((agg_type_info.is_integer() || agg_type_info.is_time()) ? "agg_sum" : "agg_sum_double");
        break;
      }
      case kCOUNT:
        result.push_back(agg_expr->get_is_distinct() ? "agg_count_distinct" : "agg_count");
        break;
      case kAPPROX_COUNT_DISTINCT:
        result.push_back("agg_approximate_count_distinct");
        break;
      default:
        CHECK(false);
    }
  }
  return result;
}

}  // namespace

std::unordered_set<llvm::Function*> Executor::markDeadRuntimeFuncs(llvm::Module& module,
                                                                   const std::vector<llvm::Function*>& roots,
                                                                   const std::vector<llvm::Function*>& leaves) {
  std::unordered_set<llvm::Function*> live_funcs;
  live_funcs.insert(roots.begin(), roots.end());
  live_funcs.insert(leaves.begin(), leaves.end());

  if (auto F = module.getFunction("init_shared_mem_nop"))
    live_funcs.insert(F);
  if (auto F = module.getFunction("write_back_nop"))
    live_funcs.insert(F);

  for (const llvm::Function* F : roots) {
    for (const llvm::BasicBlock& BB : *F) {
      for (const llvm::Instruction& I : BB) {
        if (const llvm::CallInst* CI = llvm::dyn_cast<const llvm::CallInst>(&I)) {
          live_funcs.insert(CI->getCalledFunction());
        }
      }
    }
  }

  for (llvm::Function& F : module) {
    if (!live_funcs.count(&F) && !F.isDeclaration()) {
      F.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
  }

  return live_funcs;
}

void Executor::createErrorCheckControlFlow(llvm::Function* query_func, bool run_with_dynamic_watchdog) {
  // check whether the row processing was successful; currently, it can
  // fail by running out of group by buffer slots
  bool done_splitting = false;
  for (auto bb_it = query_func->begin(); bb_it != query_func->end() && !done_splitting; ++bb_it) {
    llvm::Value* pos = nullptr;
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      if (run_with_dynamic_watchdog && llvm::isa<llvm::PHINode>(*inst_it)) {
        if (inst_it->getName() == "pos") {
          pos = &*inst_it;
        }
        continue;
      }
      if (!llvm::isa<llvm::CallInst>(*inst_it)) {
        continue;
      }
      auto& filter_call = llvm::cast<llvm::CallInst>(*inst_it);
      if (std::string(filter_call.getCalledFunction()->getName()) == unique_name("row_process", is_nested_)) {
        auto next_inst_it = inst_it;
        ++next_inst_it;
        auto new_bb = bb_it->splitBasicBlock(next_inst_it);
        auto& br_instr = bb_it->back();
        llvm::IRBuilder<> ir_builder(&br_instr);
        llvm::Value* err_lv = &*inst_it;
        if (run_with_dynamic_watchdog) {
          CHECK(pos);
          // run watchdog after every 64 rows
          auto and_lv = ir_builder.CreateAnd(pos, uint64_t(0x3f));
          auto call_watchdog_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_EQ, and_lv, ll_int(int64_t(0LL)));

          auto error_check_bb = bb_it->splitBasicBlock(llvm::BasicBlock::iterator(br_instr), ".error_check");
          auto& watchdog_br_instr = bb_it->back();

          auto watchdog_check_bb =
              llvm::BasicBlock::Create(cgen_state_->context_, ".watchdog_check", query_func, error_check_bb);
          llvm::IRBuilder<> watchdog_ir_builder(watchdog_check_bb);
          auto detected_timeout =
              watchdog_ir_builder.CreateCall(cgen_state_->module_->getFunction("dynamic_watchdog"), {});
          auto timeout_err_lv =
              watchdog_ir_builder.CreateSelect(detected_timeout, ll_int(Executor::ERR_OUT_OF_TIME), err_lv);
          watchdog_ir_builder.CreateBr(error_check_bb);

          llvm::ReplaceInstWithInst(&watchdog_br_instr,
                                    llvm::BranchInst::Create(watchdog_check_bb, error_check_bb, call_watchdog_lv));
          ir_builder.SetInsertPoint(&br_instr);
          auto unified_err_lv = ir_builder.CreatePHI(err_lv->getType(), 2);

          unified_err_lv->addIncoming(timeout_err_lv, watchdog_check_bb);
          unified_err_lv->addIncoming(err_lv, &*bb_it);
          err_lv = unified_err_lv;
        }
        const auto error_code_arg = get_arg_by_name(query_func, "error_code");
        err_lv = ir_builder.CreateCall(cgen_state_->module_->getFunction("record_error_code"),
                                       std::vector<llvm::Value*>{err_lv, error_code_arg});
        err_lv = ir_builder.CreateICmp(llvm::ICmpInst::ICMP_NE, err_lv, ll_int(int32_t(0)));
        auto error_bb = llvm::BasicBlock::Create(cgen_state_->context_, ".error_exit", query_func, new_bb);
        llvm::ReturnInst::Create(cgen_state_->context_, error_bb);
        llvm::ReplaceInstWithInst(&br_instr, llvm::BranchInst::Create(error_bb, new_bb, err_lv));
        done_splitting = true;
        break;
      }
    }
  }
  CHECK(done_splitting);
}

Executor::CompilationResult Executor::compileWorkUnit(const bool render_output,
                                                      const std::vector<InputTableInfo>& query_infos,
                                                      const RelAlgExecutionUnit& ra_exe_unit,
                                                      const CompilationOptions& co,
                                                      const ExecutionOptions& eo,
                                                      const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                                      const bool allow_lazy_fetch,
                                                      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                                      const size_t max_groups_buffer_entry_guess,
                                                      const size_t small_groups_buffer_entry_count,
                                                      const int8_t crt_min_byte_width,
                                                      const JoinInfo& join_info,
                                                      const bool has_cardinality_estimation) {
  nukeOldState(allow_lazy_fetch, join_info, query_infos, ra_exe_unit.outer_join_quals);

  GroupByAndAggregate group_by_and_aggregate(this,
                                             co.device_type_,
                                             ra_exe_unit,
                                             render_output,
                                             query_infos,
                                             row_set_mem_owner,
                                             max_groups_buffer_entry_guess,
                                             small_groups_buffer_entry_count,
                                             crt_min_byte_width,
                                             eo.allow_multifrag,
                                             eo.output_columnar_hint && co.device_type_ == ExecutorDeviceType::GPU);
  const auto& query_mem_desc = group_by_and_aggregate.getQueryMemoryDescriptor();

  if (query_mem_desc.hash_type == GroupByColRangeType::MultiCol && !query_mem_desc.getSmallBufferSizeBytes() &&
      !has_cardinality_estimation && !render_output && !eo.just_explain) {
    throw CardinalityEstimationRequired();
  }

  const bool output_columnar = group_by_and_aggregate.outputColumnar();

  if (co.device_type_ == ExecutorDeviceType::GPU) {
    for (const auto& count_distinct_descriptor : query_mem_desc.count_distinct_descriptors_) {
      if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::StdSet ||
          (count_distinct_descriptor.impl_type_ != CountDistinctImplType::Invalid && !co.hoist_literals_)) {
        throw QueryMustRunOnCpu();
      }
    }
  }

  if (co.device_type_ == ExecutorDeviceType::GPU &&
      query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash) {
    const auto grid_size = query_mem_desc.blocksShareMemory() ? 1 : gridSize();
    const size_t required_memory{(grid_size * query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU))};
    CHECK(catalog_->get_dataMgr().cudaMgr_);
    const size_t max_memory{catalog_->get_dataMgr().cudaMgr_->deviceProperties[0].globalMem / 5};
    if (required_memory > max_memory) {
      throw QueryMustRunOnCpu();
    }
  }

  // Read the module template and target either CPU or GPU
  // by binding the stream position functions to the right implementation:
  // stride access for GPU, contiguous for CPU
  cgen_state_->module_ = read_template_module(cgen_state_->context_);

  auto agg_fnames = get_agg_fnames(ra_exe_unit.target_exprs, !ra_exe_unit.groupby_exprs.empty());
  const auto agg_slot_count = ra_exe_unit.estimator ? size_t(1) : agg_fnames.size();

  const bool is_group_by{!query_mem_desc.group_col_widths.empty()};
  auto query_func =
      is_group_by ? query_group_by_template(cgen_state_->module_,
                                            is_nested_,
                                            co.hoist_literals_,
                                            query_mem_desc,
                                            co.device_type_,
                                            ra_exe_unit.scan_limit)
                  : query_template(
                        cgen_state_->module_, agg_slot_count, is_nested_, co.hoist_literals_, !!ra_exe_unit.estimator);
  bind_pos_placeholders("pos_start", true, query_func, cgen_state_->module_);
  bind_pos_placeholders("group_buff_idx", false, query_func, cgen_state_->module_);
  bind_pos_placeholders("pos_step", false, query_func, cgen_state_->module_);

  std::vector<llvm::Value*> col_heads;
  std::tie(cgen_state_->row_func_, col_heads) = create_row_function(ra_exe_unit.input_col_descs.size(),
                                                                    is_group_by ? 0 : agg_slot_count,
                                                                    co.hoist_literals_,
                                                                    query_func,
                                                                    cgen_state_->module_,
                                                                    cgen_state_->context_);
  CHECK(cgen_state_->row_func_);

  // make sure it's in-lined, we don't want register spills in the inner loop
  cgen_state_->row_func_->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);

  auto bb = llvm::BasicBlock::Create(cgen_state_->context_, "entry", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(bb);
  preloadFragOffsets(ra_exe_unit.input_descs, query_infos);

  RelAlgExecutionUnit body_execution_unit = ra_exe_unit;
  const auto join_loops = buildJoinLoops(body_execution_unit, co, eo, query_infos);

  allocateLocalColumnIds(ra_exe_unit.input_col_descs);
  if (!join_loops.empty()) {
    codegenJoinLoops(join_loops, body_execution_unit, group_by_and_aggregate, query_func, bb, co, eo);
  } else {
    const bool can_return_error = compileBody(ra_exe_unit, group_by_and_aggregate, co);

    if (can_return_error || cgen_state_->needs_error_check_ || eo.with_dynamic_watchdog) {
      createErrorCheckControlFlow(query_func, eo.with_dynamic_watchdog);
    }
  }

  // iterate through all the instruction in the query template function and
  // replace the call to the filter placeholder with the call to the actual filter
  for (auto it = llvm::inst_begin(query_func), e = llvm::inst_end(query_func); it != e; ++it) {
    if (!llvm::isa<llvm::CallInst>(*it)) {
      continue;
    }
    auto& filter_call = llvm::cast<llvm::CallInst>(*it);
    if (std::string(filter_call.getCalledFunction()->getName()) == unique_name("row_process", is_nested_)) {
      std::vector<llvm::Value*> args;
      for (size_t i = 0; i < filter_call.getNumArgOperands(); ++i) {
        args.push_back(filter_call.getArgOperand(i));
      }
      args.insert(args.end(), col_heads.begin(), col_heads.end());
      args.push_back(get_arg_by_name(query_func, "join_hash_tables"));
      llvm::ReplaceInstWithInst(&filter_call, llvm::CallInst::Create(cgen_state_->row_func_, args, ""));
      break;
    }
  }

  is_nested_ = false;
  plan_state_->init_agg_vals_ = init_agg_val_vec(ra_exe_unit.target_exprs, ra_exe_unit.quals, query_mem_desc);

  auto multifrag_query_func =
      cgen_state_->module_->getFunction("multifrag_query" + std::string(co.hoist_literals_ ? "_hoisted_literals" : ""));
  CHECK(multifrag_query_func);

  bind_query(query_func,
             "query_stub" + std::string(co.hoist_literals_ ? "_hoisted_literals" : ""),
             multifrag_query_func,
             cgen_state_->module_);

  auto live_funcs =
      markDeadRuntimeFuncs(*cgen_state_->module_, {query_func, cgen_state_->row_func_}, {multifrag_query_func});

  std::string llvm_ir;
  if (eo.just_explain) {
    llvm_ir = serialize_llvm_object(query_func) + serialize_llvm_object(cgen_state_->row_func_);
  }
  verify_function_ir(cgen_state_->row_func_);
  return Executor::CompilationResult{
      co.device_type_ == ExecutorDeviceType::CPU
          ? optimizeAndCodegenCPU(query_func, multifrag_query_func, live_funcs, cgen_state_->module_, co)
          : optimizeAndCodegenGPU(query_func,
                                  multifrag_query_func,
                                  live_funcs,
                                  cgen_state_->module_,
                                  is_group_by || ra_exe_unit.estimator,
                                  cuda_mgr,
                                  co),
      cgen_state_->getLiterals(),
      query_mem_desc,
      output_columnar,
      llvm_ir};
}

bool Executor::compileBody(const RelAlgExecutionUnit& ra_exe_unit,
                           GroupByAndAggregate& group_by_and_aggregate,
                           const CompilationOptions& co) {
  // generate the code for the filter
  std::vector<Analyzer::Expr*> primary_quals;
  std::vector<Analyzer::Expr*> deferred_quals;
  bool short_circuited = prioritizeQuals(ra_exe_unit, primary_quals, deferred_quals);
  if (short_circuited) {
    VLOG(1) << "Prioritized " << std::to_string(primary_quals.size()) << " quals, "
            << "short-circuited and deferred " << std::to_string(deferred_quals.size()) << " quals";
  }

  primary_quals = codegenHashJoinsBeforeLoopJoin(primary_quals, ra_exe_unit, co);

  if (ra_exe_unit.inner_joins.empty()) {
    allocateInnerScansIterators(ra_exe_unit.input_descs);
  }

  llvm::Value* outer_join_nomatch_flag_lv = nullptr;
  if (isOuterJoin()) {
    if (isOuterLoopJoin()) {
      CHECK(cgen_state_->outer_join_nomatch_);
      outer_join_nomatch_flag_lv = cgen_state_->ir_builder_.CreateLoad(cgen_state_->outer_join_nomatch_);
      cgen_state_->outer_join_cond_lv_ = cgen_state_->ir_builder_.CreateNot(outer_join_nomatch_flag_lv);
    } else {
      cgen_state_->outer_join_cond_lv_ = ll_bool(true);
    }
    for (auto expr : ra_exe_unit.outer_join_quals) {
      cgen_state_->outer_join_cond_lv_ = cgen_state_->ir_builder_.CreateAnd(
          cgen_state_->outer_join_cond_lv_, toBool(codegen(expr.get(), true, co).front()));
    }
    if (isOneToManyOuterHashJoin()) {
      CHECK(cgen_state_->outer_join_nomatch_);
      // TODO(miyu): Support more than 1 one-to-many hash joins in folded sequence.
      outer_join_nomatch_flag_lv = cgen_state_->ir_builder_.CreateLoad(cgen_state_->outer_join_nomatch_);
    }
  }

  llvm::Value* filter_lv =
      isOuterLoopJoin() || isOneToManyOuterHashJoin() ? cgen_state_->outer_join_cond_lv_ : ll_bool(true);
  llvm::Value* outerjoin_query_filter_lv =
      (!primary_quals.empty() && (isOuterJoin() || isOuterLoopJoin())) ? ll_bool(true) : nullptr;
  for (auto expr : primary_quals) {
    // Generate the filter for primary quals
    auto cond = toBool(codegen(expr, true, co).front());
    auto new_cond = codegenRetOnHashFail(cond, expr);
    if (new_cond == cond && !outerjoin_query_filter_lv) {
      filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, cond);
    }
    if (outerjoin_query_filter_lv) {
      outerjoin_query_filter_lv = cgen_state_->ir_builder_.CreateAnd(outerjoin_query_filter_lv, cond);
    }
  }
  CHECK(filter_lv->getType()->isIntegerTy(1));

  llvm::BasicBlock* sc_false{nullptr};
  if (!deferred_quals.empty()) {
    auto sc_true = llvm::BasicBlock::Create(cgen_state_->context_, "sc_true", cgen_state_->row_func_);
    sc_false = llvm::BasicBlock::Create(cgen_state_->context_, "sc_false", cgen_state_->row_func_);
    if (isOuterLoopJoin() || isOneToManyOuterHashJoin()) {
      filter_lv = cgen_state_->ir_builder_.CreateOr(filter_lv, outer_join_nomatch_flag_lv);
    }
    cgen_state_->ir_builder_.CreateCondBr(filter_lv, sc_true, sc_false);
    cgen_state_->ir_builder_.SetInsertPoint(sc_false);
    if (ra_exe_unit.inner_joins.empty()) {
      codegenInnerScanNextRowOrMatch();
    }
    cgen_state_->ir_builder_.SetInsertPoint(sc_true);
    filter_lv = ll_bool(true);
  }

  for (auto expr : deferred_quals) {
    filter_lv = cgen_state_->ir_builder_.CreateAnd(filter_lv, toBool(codegen(expr, true, co).front()));
  }
  if (isOuterLoopJoin() || isOneToManyOuterHashJoin()) {
    CHECK(outer_join_nomatch_flag_lv);
    filter_lv = cgen_state_->ir_builder_.CreateOr(filter_lv, outer_join_nomatch_flag_lv);
  }

  CHECK(filter_lv->getType()->isIntegerTy(1));

  return group_by_and_aggregate.codegen(filter_lv, outerjoin_query_filter_lv, sc_false, co);
}
