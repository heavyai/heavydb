/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include "IRCodegenUtils.h"
#include "InValuesBitmap.h"
#include "InputMetadata.h"
#include "LLVMGlobalContext.h"
#include "StringDictionaryTranslationMgr.h"
#include "TreeModelPredictionMgr.h"

#include "../Analyzer/Analyzer.h"
#include "../Shared/InsertionOrderedMap.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include "Shared/DbObjectKeys.h"

struct ArrayLoadCodegen {
  llvm::Value* buffer;
  llvm::Value* size;
  llvm::Value* is_null;
};

struct CgenState {
 public:
  CgenState(const size_t num_query_infos,
            const bool contains_left_deep_outer_join,
            Executor* executor);
  CgenState(const size_t num_query_infos, const bool contains_left_deep_outer_join);
  CgenState(llvm::LLVMContext& context);

  std::tuple<size_t, size_t> getOrAddLiteral(const Analyzer::Constant* constant,
                                             const EncodingType enc_type,
                                             const shared::StringDictKey& dict_id,
                                             const int device_id) {
    const auto& ti = constant->get_type_info();
    const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
    switch (type) {
      case kBOOLEAN:
        return getOrAddLiteral(constant->get_is_null()
                                   ? int8_t(inline_int_null_val(ti))
                                   : int8_t(constant->get_constval().boolval ? 1 : 0),
                               device_id);
      case kTINYINT:
        return getOrAddLiteral(constant->get_is_null()
                                   ? int8_t(inline_int_null_val(ti))
                                   : constant->get_constval().tinyintval,
                               device_id);
      case kSMALLINT:
        return getOrAddLiteral(constant->get_is_null()
                                   ? int16_t(inline_int_null_val(ti))
                                   : constant->get_constval().smallintval,
                               device_id);
      case kINT:
        return getOrAddLiteral(constant->get_is_null() ? int32_t(inline_int_null_val(ti))
                                                       : constant->get_constval().intval,
                               device_id);
      case kBIGINT:
        return getOrAddLiteral(constant->get_is_null()
                                   ? int64_t(inline_int_null_val(ti))
                                   : constant->get_constval().bigintval,
                               device_id);
      case kFLOAT:
        return getOrAddLiteral(constant->get_is_null()
                                   ? float(inline_fp_null_val(ti))
                                   : constant->get_constval().floatval,
                               device_id);
      case kDOUBLE:
        return getOrAddLiteral(constant->get_is_null()
                                   ? inline_fp_null_val(ti)
                                   : constant->get_constval().doubleval,
                               device_id);
      case kCHAR:
      case kTEXT:
      case kVARCHAR:
        if (enc_type == kENCODING_DICT) {
          if (constant->get_is_null()) {
            return getOrAddLiteral(int32_t(inline_int_null_val(ti)), device_id);
          }
          return getOrAddLiteral(
              std::make_pair(*constant->get_constval().stringval, dict_id), device_id);
        }
        CHECK_EQ(kENCODING_NONE, enc_type);
        if (constant->get_is_null()) {
          throw std::runtime_error(
              "CHAR / VARCHAR NULL literal not supported in this context");  // TODO(alex):
                                                                             // support
                                                                             // null
        }
        return getOrAddLiteral(*constant->get_constval().stringval, device_id);
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
        // TODO(alex): support null
        return getOrAddLiteral(constant->get_constval().bigintval, device_id);
      case kARRAY: {
        if (enc_type == kENCODING_NONE) {
          if (ti.get_subtype() == kDOUBLE) {
            std::vector<double> double_array_literal;
            for (const auto& value : constant->get_value_list()) {
              const auto c = dynamic_cast<const Analyzer::Constant*>(value.get());
              CHECK(c);
              double d = c->get_constval().doubleval;
              double_array_literal.push_back(d);
            }
            return getOrAddLiteral(double_array_literal, device_id);
          }
          if (ti.get_subtype() == kINT) {
            std::vector<int32_t> int32_array_literal;
            for (const auto& value : constant->get_value_list()) {
              const auto c = dynamic_cast<const Analyzer::Constant*>(value.get());
              CHECK(c);
              int32_t i = c->get_constval().intval;
              int32_array_literal.push_back(i);
            }
            return getOrAddLiteral(int32_array_literal, device_id);
          }
          if (ti.get_subtype() == kTINYINT) {
            std::vector<int8_t> int8_array_literal;
            for (const auto& value : constant->get_value_list()) {
              const auto c = dynamic_cast<const Analyzer::Constant*>(value.get());
              CHECK(c);
              int8_t i = c->get_constval().tinyintval;
              int8_array_literal.push_back(i);
            }
            if (ti.get_comp_param() == 64) {
              return getOrAddLiteral(std::make_pair(int8_array_literal, 64), device_id);
            }
            return getOrAddLiteral(int8_array_literal, device_id);
          }
          throw std::runtime_error("Unsupported literal array");
        }
        if (enc_type == kENCODING_GEOINT) {
          if (ti.get_subtype() == kTINYINT) {
            std::vector<int8_t> int8_array_literal;
            for (const auto& value : constant->get_value_list()) {
              const auto c = dynamic_cast<const Analyzer::Constant*>(value.get());
              CHECK(c);
              int8_t i = c->get_constval().tinyintval;
              int8_array_literal.push_back(i);
            }
            if (ti.get_comp_param() == 32) {
              return getOrAddLiteral(std::make_pair(int8_array_literal, 32), device_id);
            }
            return getOrAddLiteral(int8_array_literal, device_id);
          }
        }
        throw std::runtime_error("Encoded literal arrays are not supported");
      }
      default:
        abort();
    }
  }

  using LiteralValue = boost::variant<int8_t,
                                      int16_t,
                                      int32_t,
                                      int64_t,
                                      float,
                                      double,
                                      std::pair<std::string, shared::StringDictKey>,
                                      std::string,
                                      std::vector<double>,
                                      std::vector<int32_t>,
                                      std::vector<int8_t>,
                                      std::pair<std::vector<int8_t>, int>>;
  using LiteralValues = std::vector<LiteralValue>;

  const std::unordered_map<int, LiteralValues>& getLiterals() const { return literals_; }

  llvm::Value* addStringConstant(const std::string& str) {
    llvm::Value* str_lv = ir_builder_.CreateGlobalString(
        str, "str_const_" + std::to_string(std::hash<std::string>()(str)));
    auto i8_ptr = llvm::PointerType::get(get_int_type(8, context_), 0);
    str_constants_.push_back(str_lv);
    str_lv = ir_builder_.CreateBitCast(str_lv, i8_ptr);
    return str_lv;
  }

  const StringDictionaryTranslationMgr* moveStringDictionaryTranslationMgr(
      std::unique_ptr<const StringDictionaryTranslationMgr>&& str_dict_translation_mgr) {
    str_dict_translation_mgrs_.emplace_back(std::move(str_dict_translation_mgr));
    return str_dict_translation_mgrs_.back().get();
  }

  const TreeModelPredictionMgr* moveTreeModelPredictionMgr(
      std::unique_ptr<const TreeModelPredictionMgr>&& tree_model_prediction_mgr) {
    tree_model_prediction_mgrs_.emplace_back(std::move(tree_model_prediction_mgr));
    return tree_model_prediction_mgrs_.back().get();
  }

  const InValuesBitmap* addInValuesBitmap(
      std::unique_ptr<InValuesBitmap>& in_values_bitmap) {
    if (in_values_bitmap->isEmpty()) {
      return in_values_bitmap.get();
    }
    in_values_bitmaps_.emplace_back(std::move(in_values_bitmap));
    return in_values_bitmaps_.back().get();
  }
  void moveInValuesBitmap(std::unique_ptr<const InValuesBitmap>& in_values_bitmap) {
    if (!in_values_bitmap->isEmpty()) {
      in_values_bitmaps_.emplace_back(std::move(in_values_bitmap));
    }
  }
  // look up a runtime function based on the name, return type and type of
  // the arguments and call it; x64 only, don't call from GPU codegen
  llvm::Value* emitExternalCall(
      const std::string& fname,
      llvm::Type* ret_type,
      const std::vector<llvm::Value*> args,
      const std::vector<llvm::Attribute::AttrKind>& fnattrs = {},
      const bool has_struct_return = false);
  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);
  llvm::Value* emitEntryCall(const std::string& fname,
                             const std::vector<llvm::Value*>& args);

  size_t getLiteralBufferUsage(const int device_id) { return literal_bytes_[device_id]; }

  llvm::Value* castToTypeIn(llvm::Value* val, const size_t bit_width);

  std::pair<llvm::ConstantInt*, llvm::ConstantInt*> inlineIntMaxMin(
      const size_t byte_width,
      const bool is_signed);

  llvm::ConstantInt* inlineIntNull(const SQLTypeInfo&);
  llvm::ConstantFP* inlineFpNull(const SQLTypeInfo&);
  llvm::Constant* inlineNull(const SQLTypeInfo&);

  template <class T>
  llvm::ConstantInt* llInt(const T v) const {
    return ::ll_int(v, context_);
  }

  llvm::ConstantFP* llFp(const float v) const {
    return static_cast<llvm::ConstantFP*>(
        llvm::ConstantFP::get(llvm::Type::getFloatTy(context_), v));
  }

  llvm::ConstantFP* llFp(const double v) const {
    return static_cast<llvm::ConstantFP*>(
        llvm::ConstantFP::get(llvm::Type::getDoubleTy(context_), v));
  }

  llvm::ConstantInt* llBool(const bool v) const { return ::ll_bool(v, context_); }

  void emitErrorCheck(llvm::Value* condition, llvm::Value* errorCode, std::string label);

  std::vector<std::string> gpuFunctionsToReplace(llvm::Function* fn);

  void replaceFunctionForGpu(const std::string& fcn_to_replace, llvm::Function* fn);

  std::shared_ptr<Executor> getExecutor() const;
  llvm::LLVMContext& getExecutorContext() const;
  void set_module_shallow_copy(const std::unique_ptr<llvm::Module>& module,
                               bool always_clone = false);

  size_t executor_id_;

  /*
    Managing LLVM modules
    ---------------------

    Quoting https://groups.google.com/g/llvm-dev/c/kuil5XjasUs/m/7PBpOWZFDAAJ :
    """
    The state of Module/Context ownership is very muddled in the
    codebase. As you have discovered: LLVMContext’s notionally own
    their modules (via raw pointers deleted upon destruction of the
    context), however in practice we always hold llvm Modules by
    unique_ptr. Since the Module destructor removes the raw pointer
    from the Context, this doesn’t usually blow up. It’s pretty broken
    though.

    I would argue that you should use unique_ptr and ignore
    LLVMContext ownership.
    """

    Here we follow the last argument only partially for reasons
    explained below.

    HeavyDB supports concurrent query executions. For that, a global
    cache of Executor instances is used. Each instance is able to
    generate LLVM code, compile it to machine code (with code
    caching), and execute the code --- all that concurrently with
    other Executor instances.

    Each Executor instance holds as set of extension modules (LLVM
    Module instances) that are either loaded at Executor construction
    time (template module from RuntimeFunctions.bc, rt_geos from
    GeosRuntime.bc, rt_libdevice from libdevice.10.bc, udf_cpu/gpu
    modules from LLVM IR file), or at run-time (rt_udf_cpu/gpu modules
    from LLVM IR string).  All these extension modules are owned by
    the Executor instance via unique_ptr. Since Executor also owns the
    LLVM Context instance that technically also owns these extension
    modules, then the LLVM Context-Module ownership can be ignored
    (see the quote above).

    Code generation is a process that compiles
    (generated/user-provided) LLVM IR code into machine code that can
    be executed on a CPU or GPU.

    Typically, a copy of the template module (let's call this copy as
    a worker module) is used as an input to code generation that is
    updated with generated/user-provided LLVM Functions and with other
    extension modules being linked in. The worker module is created by
    set_module_shallow_copy and is owned by an Executor instance as a
    raw pointer (via cgen_state member). Notice that
    set_module_shallow_copy clones the template module and then
    releases unique_ptr as a raw pointer.  This means that Executor is
    now responsible of deleting the worker module after the
    compilation process completes.

    The reason why the worker module is stored via raw pointer value
    (rather than using unique_ptr as suggested in the quote above) is
    as follows.  First, the code generation in HeavyDB can be a
    recursive process (e.g. in the case of multi-step
    multi-subqueries) that involves temporary "shelving" of parent
    compilation processes (the corresponding worker modules are put on
    hold). In addition, the Executor may trigger threaded compilations
    that involve "resetting" the worker module for different threads
    (btw, these compilations cannot be concurrent because LLVM Context
    is not threadsafe.  The shelving and resetting of worker modules
    makes the scope of a worker module dynamic (only one worker module
    instance can be in scope while other worker modules are on hold)
    that contradicts with the purpose of unique_ptr (out-of-scope
    worker modules can be destroyed) and would make managing all
    worker modules very painful if these would be stored as unique_ptr
    instances.

    An entry point to the code generation is Executor::compileWorkUnit
    method. Its scope includes creating an Executor::CgenStateManager
    instance that uses RAII pattern to manage the CgenState instance
    held by an Executor instance. In addition, the CgenStateManager
    locks other compilations within the same Executor instance. The
    compilation lock enables the threaded compilation feature.

    Construction of CgenStateManager (i) stores the existing CgenState
    instance held by the Executor instance, and (ii) creates an new
    CgenState instance with un-instantiated worker module.  The worker
    module is instantiated after the construction (unless
    QueryMustRunOnCpu is thrown) via set_module_shallow_copy, followed
    by updating the worker module according to the given query and
    compiling it to machine code. Destruction of CgenStateManager
    (read: when leaving the compileWorkUnit method) will delete the
    instantiated worker module and restores the previous CgenState
    instance.  This CgenState management enables the recursive
    compilation feature.

    Finally, we note that the worker module compilation caches the
    compilation results using the full LLVM IR as the cache
    key. Caching compilation results is especially effective for CUDA
    target due to a considerable overhead from the CUDA compilation.
   */

  llvm::Module* module_;
  llvm::Function* row_func_;
  llvm::Function* filter_func_;
  llvm::Function* current_func_;
  llvm::BasicBlock* row_func_bb_;
  llvm::BasicBlock* filter_func_bb_;
  llvm::CallInst* row_func_call_;
  llvm::CallInst* filter_func_call_;
  std::vector<llvm::Function*> helper_functions_;
  llvm::LLVMContext& context_;    // LLVMContext instance is held by an Executor instance.
  llvm::ValueToValueMapTy vmap_;  // used for cloning the runtime module
  llvm::IRBuilder<> ir_builder_;
  std::unordered_map<size_t, std::vector<llvm::Value*>> fetch_cache_;
  struct FunctionOperValue {
    const Analyzer::FunctionOper* foper;
    llvm::Value* lv;
  };
  std::vector<FunctionOperValue> ext_call_cache_;
  std::vector<llvm::Value*> group_by_expr_cache_;
  std::vector<llvm::Value*> str_constants_;
  std::vector<llvm::Value*> frag_offsets_;
  const bool contains_left_deep_outer_join_;
  std::vector<llvm::Value*> outer_join_match_found_per_level_;
  std::unordered_map<int, llvm::Value*> scan_idx_to_hash_pos_;
  InsertionOrderedMap filter_func_args_;
  std::vector<std::unique_ptr<const InValuesBitmap>> in_values_bitmaps_;
  std::vector<std::unique_ptr<const TreeModelPredictionMgr>> tree_model_prediction_mgrs_;
  std::vector<std::unique_ptr<const StringDictionaryTranslationMgr>>
      str_dict_translation_mgrs_;
  std::map<std::pair<llvm::Value*, llvm::Value*>, ArrayLoadCodegen>
      array_load_cache_;  // byte stream to array info
  std::unordered_map<std::string, llvm::Value*> geo_target_cache_;
  bool needs_error_check_;
  bool needs_geos_;

  llvm::Function* query_func_;
  llvm::IRBuilder<> query_func_entry_ir_builder_;
  std::unordered_map<int, std::vector<llvm::Value*>> query_func_literal_loads_;

  struct HoistedLiteralLoadLocator {
    int offset_in_literal_buffer;
    int index_of_literal_load;
  };
  std::unordered_map<llvm::Value*, HoistedLiteralLoadLocator> row_func_hoisted_literals_;

  static size_t literalBytes(const CgenState::LiteralValue& lit) {
    switch (lit.which()) {
      case 0:
        return 1;  // int8_t
      case 1:
        return 2;  // int16_t
      case 2:
        return 4;  // int32_t
      case 3:
        return 8;  // int64_t
      case 4:
        return 4;  // float
      case 5:
        return 8;  // double
      case 6:
        return 4;  // std::pair<std::string, int>
      case 7:
        return 4;  // std::string
      case 8:
        return 4;  // std::vector<double>
      case 9:
        return 4;  // std::vector<int32_t>
      case 10:
        return 4;  // std::vector<int8_t>
      case 11:
        return 4;  // std::pair<std::vector<int8_t>, int>
      default:
        abort();
    }
  }

  static size_t addAligned(const size_t off_in, const size_t alignment) {
    size_t off = off_in;
    if (off % alignment != 0) {
      off += (alignment - off % alignment);
    }
    return off + alignment;
  }

  void maybeCloneFunctionRecursive(llvm::Function* fn);

 private:
  // todo (yoonmin) : avoid linear scanning of `literals` map
  template <class T>
  std::tuple<size_t, size_t> getOrAddLiteral(const T& val, const int device_id) {
    const LiteralValue var_val(val);
    size_t literal_found_off{0};
    auto& literals = literals_[device_id];
    for (const auto& literal : literals) {
      const auto lit_bytes = literalBytes(literal);
      literal_found_off = addAligned(literal_found_off, lit_bytes);
      if (literal == var_val) {
        return {literal_found_off - lit_bytes, lit_bytes};
      }
    }
    literals.emplace_back(val);
    const auto lit_bytes = literalBytes(var_val);
    literal_bytes_[device_id] = addAligned(literal_bytes_[device_id], lit_bytes);
    return {literal_bytes_[device_id] - lit_bytes, lit_bytes};
  }

  std::unordered_map<int, LiteralValues> literals_;
  std::unordered_map<int, size_t> literal_bytes_;
};

#include "AutomaticIRMetadataGuard.h"
