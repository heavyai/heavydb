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

#include "QueryEngine/CgenState.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/OutputBufferInitialization.h"

#include <llvm/IR/InstIterator.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

CgenState::CgenState(const size_t num_query_infos,
                     const bool contains_left_deep_outer_join,
                     Executor* executor)
    : executor_id_(executor->getExecutorId())
    , module_(nullptr)
    , row_func_(nullptr)
    , filter_func_(nullptr)
    , current_func_(nullptr)
    , row_func_bb_(nullptr)
    , filter_func_bb_(nullptr)
    , row_func_call_(nullptr)
    , filter_func_call_(nullptr)
    , context_(executor->getContext())
    , ir_builder_(context_)
    , contains_left_deep_outer_join_(contains_left_deep_outer_join)
    , outer_join_match_found_per_level_(std::max(num_query_infos, size_t(1)) - 1)
    , needs_error_check_(false)
    , automatic_ir_metadata_(executor->getConfig().debug.enable_automatic_ir_metadata)
    , query_func_(nullptr)
    , query_func_entry_ir_builder_(context_){};

// scalar code generator constructor
CgenState::CgenState(const size_t num_query_infos,
                     const bool contains_left_deep_outer_join)
    : CgenState(
          num_query_infos,
          contains_left_deep_outer_join,
          Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get()) {}

CgenState::CgenState(const Config& config, llvm::LLVMContext& context)
    : executor_id_(Executor::INVALID_EXECUTOR_ID)
    , module_(nullptr)
    , row_func_(nullptr)
    , context_(context)
    , ir_builder_(context_)
    , contains_left_deep_outer_join_(false)
    , needs_error_check_(false)
    , automatic_ir_metadata_(config.debug.enable_automatic_ir_metadata)
    , query_func_(nullptr)
    , query_func_entry_ir_builder_(context_){};

llvm::ConstantInt* CgenState::inlineIntNull(const SQLTypeInfo& type_info) {
  auto type = type_info.get_type();
  if (type_info.is_string()) {
    switch (type_info.get_compression()) {
      case kENCODING_DICT:
        return llInt(static_cast<int32_t>(inline_int_null_val(type_info)));
      case kENCODING_NONE:
        return llInt(int64_t(0));
      default:
        CHECK(false);
    }
  }
  switch (type) {
    case kBOOLEAN:
      return llInt(static_cast<int8_t>(inline_int_null_val(type_info)));
    case kTINYINT:
      return llInt(static_cast<int8_t>(inline_int_null_val(type_info)));
    case kSMALLINT:
      return llInt(static_cast<int16_t>(inline_int_null_val(type_info)));
    case kINT:
      return llInt(static_cast<int32_t>(inline_int_null_val(type_info)));
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return llInt(inline_int_null_val(type_info));
    case kDECIMAL:
    case kNUMERIC:
      return llInt(inline_int_null_val(type_info));
    case kARRAY:
      return llInt(int64_t(0));
    default:
      abort();
  }
}

llvm::ConstantFP* CgenState::inlineFpNull(const SQLTypeInfo& type_info) {
  CHECK(type_info.is_fp());
  switch (type_info.get_type()) {
    case kFLOAT:
      return llFp(NULL_FLOAT);
    case kDOUBLE:
      return llFp(NULL_DOUBLE);
    default:
      abort();
  }
}

llvm::Constant* CgenState::inlineNull(const SQLTypeInfo& ti) {
  return ti.is_fp() ? static_cast<llvm::Constant*>(inlineFpNull(ti))
                    : static_cast<llvm::Constant*>(inlineIntNull(ti));
}

std::pair<llvm::ConstantInt*, llvm::ConstantInt*> CgenState::inlineIntMaxMin(
    const size_t byte_width,
    const bool is_signed) {
  int64_t max_int{0}, min_int{0};
  if (is_signed) {
    std::tie(max_int, min_int) = inline_int_max_min(byte_width);
  } else {
    uint64_t max_uint{0}, min_uint{0};
    std::tie(max_uint, min_uint) = inline_uint_max_min(byte_width);
    max_int = static_cast<int64_t>(max_uint);
    CHECK_EQ(uint64_t(0), min_uint);
  }
  switch (byte_width) {
    case 1:
      return std::make_pair(::ll_int(static_cast<int8_t>(max_int), context_),
                            ::ll_int(static_cast<int8_t>(min_int), context_));
    case 2:
      return std::make_pair(::ll_int(static_cast<int16_t>(max_int), context_),
                            ::ll_int(static_cast<int16_t>(min_int), context_));
    case 4:
      return std::make_pair(::ll_int(static_cast<int32_t>(max_int), context_),
                            ::ll_int(static_cast<int32_t>(min_int), context_));
    case 8:
      return std::make_pair(::ll_int(max_int, context_), ::ll_int(min_int, context_));
    default:
      abort();
  }
}

llvm::Value* CgenState::castToTypeIn(llvm::Value* val, const size_t dst_bits) {
  auto src_bits = val->getType()->getScalarSizeInBits();
  if (src_bits == dst_bits) {
    return val;
  }
  if (val->getType()->isIntegerTy()) {
    return ir_builder_.CreateIntCast(
        val, get_int_type(dst_bits, context_), src_bits != 1);
  }
  // real (not dictionary-encoded) strings; store the pointer to the payload
  if (val->getType()->isPointerTy()) {
    return ir_builder_.CreatePointerCast(val, get_int_type(dst_bits, context_));
  }

  CHECK(val->getType()->isFloatTy() || val->getType()->isDoubleTy());

  llvm::Type* dst_type = nullptr;
  switch (dst_bits) {
    case 64:
      dst_type = llvm::Type::getDoubleTy(context_);
      break;
    case 32:
      dst_type = llvm::Type::getFloatTy(context_);
      break;
    default:
      CHECK(false);
  }

  return ir_builder_.CreateFPCast(val, dst_type);
}

void CgenState::maybeCloneFunctionRecursive(llvm::Function* fn) {
  CHECK(fn);
  if (!fn->isDeclaration()) {
    return;
  }

  // Get the implementation from the runtime module.
  auto func_impl = getExecutor()->get_rt_module()->getFunction(fn->getName());
  CHECK(func_impl) << fn->getName().str();

  if (func_impl->isDeclaration()) {
    return;
  }

  auto DestI = fn->arg_begin();
  for (auto arg_it = func_impl->arg_begin(); arg_it != func_impl->arg_end(); ++arg_it) {
    DestI->setName(arg_it->getName());
    vmap_[&*arg_it] = &*DestI++;
  }

  llvm::SmallVector<llvm::ReturnInst*, 8> Returns;  // Ignore returns cloned.
#if LLVM_VERSION_MAJOR > 12
  llvm::CloneFunctionInto(
      fn, func_impl, vmap_, llvm::CloneFunctionChangeType::DifferentModule, Returns);
#else
  llvm::CloneFunctionInto(fn, func_impl, vmap_, /*ModuleLevelChanges=*/true, Returns);
#endif

  for (auto it = llvm::inst_begin(fn), e = llvm::inst_end(fn); it != e; ++it) {
    if (llvm::isa<llvm::CallInst>(*it)) {
      auto& call = llvm::cast<llvm::CallInst>(*it);
      maybeCloneFunctionRecursive(call.getCalledFunction());
    }
  }
}

llvm::Value* CgenState::emitCall(const std::string& fname,
                                 const std::vector<llvm::Value*>& args) {
  // Get the function reference from the query module.
  auto func = module_->getFunction(fname);
  CHECK(func);
  // If the function called isn't external, clone the implementation from the runtime
  // module.
  maybeCloneFunctionRecursive(func);

  return ir_builder_.CreateCall(func, args);
}

void CgenState::emitErrorCheck(llvm::Value* condition,
                               llvm::Value* errorCode,
                               std::string label) {
  needs_error_check_ = true;
  auto check_ok = llvm::BasicBlock::Create(context_, label + "_ok", current_func_);
  auto check_fail = llvm::BasicBlock::Create(context_, label + "_fail", current_func_);
  ir_builder_.CreateCondBr(condition, check_ok, check_fail);
  ir_builder_.SetInsertPoint(check_fail);
  ir_builder_.CreateRet(errorCode);
  ir_builder_.SetInsertPoint(check_ok);
}

namespace {

// clang-format off
template <typename T>
llvm::Type* getTy(llvm::LLVMContext& ctx) { return getTy<std::remove_pointer_t<T>>(ctx)->getPointerTo(); }
// Commented out to avoid -Wunused-function warnings, but enable as needed.
// template<> llvm::Type* getTy<bool>(llvm::LLVMContext& ctx) { return llvm::Type::getInt1Ty(ctx); }
//template<> llvm::Type* getTy<int8_t>(llvm::LLVMContext& ctx) { return llvm::Type::getInt8Ty(ctx); }
// template<> llvm::Type* getTy<int16_t>(llvm::LLVMContext& ctx) { return llvm::Type::getInt16Ty(ctx); }
//template<> llvm::Type* getTy<int32_t>(llvm::LLVMContext& ctx) { return llvm::Type::getInt32Ty(ctx); }
// template<> llvm::Type* getTy<int64_t>(llvm::LLVMContext& ctx) { return llvm::Type::getInt64Ty(ctx); }
// template<> llvm::Type* getTy<float>(llvm::LLVMContext& ctx) { return llvm::Type::getFloatTy(ctx); }
template<> llvm::Type* getTy<double>(llvm::LLVMContext& ctx) { return llvm::Type::getDoubleTy(ctx); }
//template<> llvm::Type* getTy<void>(llvm::LLVMContext& ctx) { return llvm::Type::getVoidTy(ctx); }
//  clang-format on

struct GpuFunctionDefinition {
  GpuFunctionDefinition(char const* name) : name_(name) {}
  char const* const name_;

  virtual ~GpuFunctionDefinition() = default;

  virtual llvm::FunctionCallee getFunction(llvm::Module* llvm_module,
                                           llvm::LLVMContext& context) const = 0;
};

// TYPES = return_type, arg0_type, arg1_type, arg2_type, ...
template <typename... TYPES>
struct GpuFunction final : public GpuFunctionDefinition {
  GpuFunction(char const* name) : GpuFunctionDefinition(name) {}

  llvm::FunctionCallee getFunction(llvm::Module* llvm_module,
                                   llvm::LLVMContext& context) const {
    return llvm_module->getOrInsertFunction(name_, getTy<TYPES>(context)...);
  }
};

static const std::unordered_map<std::string, std::shared_ptr<GpuFunctionDefinition>>
    gpu_replacement_functions{
        {"asin", std::make_shared<GpuFunction<double, double>>("Asin")},
        {"atanh", std::make_shared<GpuFunction<double, double>>("Atanh")},
        {"atan", std::make_shared<GpuFunction<double, double>>("Atan")},
        {"cosh", std::make_shared<GpuFunction<double, double>>("Cosh")},
        {"cos", std::make_shared<GpuFunction<double, double>>("Cos")},
        {"exp", std::make_shared<GpuFunction<double, double>>("Exp")},
        {"log", std::make_shared<GpuFunction<double, double>>("ln")},
        {"pow", std::make_shared<GpuFunction<double, double, double>>("power")},
        {"sinh", std::make_shared<GpuFunction<double, double>>("Sinh")},
        {"sin", std::make_shared<GpuFunction<double, double>>("Sin")},
        {"sqrt", std::make_shared<GpuFunction<double, double>>("Sqrt")},
        {"tan", std::make_shared<GpuFunction<double, double>>("Tan")}};
}  // namespace

std::vector<std::string> CgenState::gpuFunctionsToReplace(llvm::Function* fn) {
  std::vector<std::string> ret;

  CHECK(fn);
  CHECK(!fn->isDeclaration());

  for (auto& basic_block : *fn) {
    auto& inst_list = basic_block.getInstList();
    for (auto inst_itr = inst_list.begin(); inst_itr != inst_list.end(); ++inst_itr) {
      if (auto call_inst = llvm::dyn_cast<llvm::CallInst>(inst_itr)) {
        auto called_fcn = call_inst->getCalledFunction();
        CHECK(called_fcn);

        if (gpu_replacement_functions.find(called_fcn->getName().str()) !=
            gpu_replacement_functions.end()) {
          ret.emplace_back(called_fcn->getName());
        }
      }
    }
  }
  return ret;
}

void CgenState::replaceFunctionForGpu(const std::string& fcn_to_replace,
                                      llvm::Function* fn) {
  CHECK(fn);
  CHECK(!fn->isDeclaration());

  auto map_it = gpu_replacement_functions.find(fcn_to_replace);
  if (map_it == gpu_replacement_functions.end()) {
    throw QueryMustRunOnCpu("Codegen failed: Could not find replacement functon for " +
                            fcn_to_replace +
                            " to run on gpu. Query step must run in cpu mode.");
  }
  const auto& gpu_fcn_obj = map_it->second;
  CHECK(gpu_fcn_obj);
  VLOG(1) << "Replacing " << fcn_to_replace << " with " << gpu_fcn_obj->name_
          << " for parent function " << fn->getName().str();

  for (auto& basic_block : *fn) {
    auto& inst_list = basic_block.getInstList();
    for (auto inst_itr = inst_list.begin(); inst_itr != inst_list.end(); ++inst_itr) {
      if (auto call_inst = llvm::dyn_cast<llvm::CallInst>(inst_itr)) {
        auto called_fcn = call_inst->getCalledFunction();
        CHECK(called_fcn);

        if (called_fcn->getName() == fcn_to_replace) {
          std::vector<llvm::Value*> args;
          std::vector<llvm::Type*> arg_types;
          for (auto& arg : call_inst->args()) {
            arg_types.push_back(arg.get()->getType());
            args.push_back(arg.get());
          }
          auto gpu_func = gpu_fcn_obj->getFunction(module_, context_);
          CHECK(gpu_func);
          auto gpu_func_type = gpu_func.getFunctionType();
          CHECK(gpu_func_type);
          CHECK_EQ(gpu_func_type->getReturnType(), called_fcn->getReturnType());
          llvm::ReplaceInstWithInst(call_inst,
                                    llvm::CallInst::Create(gpu_func, args, ""));
          return;
        }
      }
    }
  }
}

std::shared_ptr<Executor> CgenState::getExecutor() const {
  CHECK(executor_id_ != Executor::INVALID_EXECUTOR_ID);
  auto executor = Executor::getExecutorFromMap(executor_id_);
  CHECK(executor);
  return executor;
}

llvm::LLVMContext& CgenState::getExecutorContext() const {
  return getExecutor()->getContext();
}

void CgenState::set_module_shallow_copy(const std::unique_ptr<llvm::Module>& llvm_module,
                                        bool always_clone) {
  module_ =
      llvm::CloneModule(*llvm_module, vmap_, [always_clone](const llvm::GlobalValue* gv) {
        auto func = llvm::dyn_cast<llvm::Function>(gv);
        if (!func) {
          return true;
        }
        return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
                func->getLinkage() == llvm::GlobalValue::LinkageTypes::InternalLinkage ||
                (always_clone && CodeGenerator::alwaysCloneRuntimeFunction(func)));
      }).release();
}
