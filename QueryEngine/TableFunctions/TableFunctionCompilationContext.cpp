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

#include "QueryEngine/TableFunctions/TableFunctionCompilationContext.h"

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/QueryEngine.h"

namespace {

llvm::Function* generate_entry_point(const CgenState* cgen_state) {
  auto& ctx = cgen_state->context_;
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto ppi8_type = llvm::PointerType::get(pi8_type, 0);
  const auto pi64_type = llvm::PointerType::get(get_int_type(64, ctx), 0);
  const auto ppi64_type = llvm::PointerType::get(pi64_type, 0);
  const auto i32_type = get_int_type(32, ctx);

  const auto func_type = llvm::FunctionType::get(
      i32_type,
      {pi8_type, ppi8_type, pi64_type, ppi8_type, ppi64_type, ppi8_type, pi64_type},
      false);

  auto func = llvm::Function::Create(func_type,
                                     llvm::Function::ExternalLinkage,
                                     "call_table_function",
                                     cgen_state->module_);
  auto arg_it = func->arg_begin();
  const auto mgr_arg = &*arg_it;
  mgr_arg->setName("mgr_ptr");
  const auto input_cols_arg = &*(++arg_it);
  input_cols_arg->setName("input_col_buffers");
  const auto input_row_counts = &*(++arg_it);
  input_row_counts->setName("input_row_counts");
  const auto input_str_dict_proxies = &*(++arg_it);
  input_str_dict_proxies->setName("input_str_dict_proxies");
  const auto output_buffers = &*(++arg_it);
  output_buffers->setName("output_buffers");
  const auto output_str_dict_proxies = &*(++arg_it);
  output_str_dict_proxies->setName("output_str_dict_proxies");
  const auto output_row_count = &*(++arg_it);
  output_row_count->setName("output_row_count");
  return func;
}

inline llvm::Type* get_llvm_type_from_sql_column_type(const SQLTypeInfo elem_ti,
                                                      llvm::LLVMContext& ctx) {
  if (elem_ti.is_fp()) {
    return get_fp_ptr_type(elem_ti.get_size() * 8, ctx);
  } else if (elem_ti.is_boolean()) {
    return get_int_ptr_type(8, ctx);
  } else if (elem_ti.is_integer()) {
    return get_int_ptr_type(elem_ti.get_size() * 8, ctx);
  } else if (elem_ti.is_string()) {
    if (elem_ti.get_compression() == kENCODING_DICT) {
      return get_int_ptr_type(elem_ti.get_size() * 8, ctx);
    }
    CHECK(elem_ti.is_text_encoding_none());
    return get_int_ptr_type(8, ctx);
  } else if (elem_ti.is_timestamp()) {
    return get_int_ptr_type(elem_ti.get_size() * 8, ctx);
  } else if (elem_ti.usesFlatBuffer()) {
    return get_int_ptr_type(8, ctx);
  }
  LOG(FATAL) << "get_llvm_type_from_sql_column_type: not implemented for "
             << ::toString(elem_ti);
  return nullptr;
}

void initialize_ptr_member(llvm::Value* member_ptr,
                           llvm::Type* member_llvm_type,
                           llvm::Value* value_ptr,
                           llvm::IRBuilder<>& ir_builder) {
  if (value_ptr != nullptr) {
    if (value_ptr->getType() == member_llvm_type->getPointerElementType()) {
      ir_builder.CreateStore(value_ptr, member_ptr);
    } else {
      auto tmp = ir_builder.CreateBitCast(value_ptr, member_llvm_type);
      ir_builder.CreateStore(tmp, member_ptr);
    }
  } else {
    ir_builder.CreateStore(llvm::Constant::getNullValue(member_llvm_type), member_ptr);
  }
}

template <typename T>
void initialize_int_member(llvm::Value* member_ptr,
                           llvm::Value* value,
                           int64_t default_value,
                           llvm::LLVMContext& ctx,
                           llvm::IRBuilder<>& ir_builder) {
  llvm::Value* val = nullptr;
  if (value != nullptr) {
    auto value_type = value->getType();
    if (value_type->isPointerTy()) {
      CHECK(value_type->getPointerElementType()->isIntegerTy(sizeof(T) * 8));
      val = ir_builder.CreateLoad(value->getType()->getPointerElementType(), value);
    } else {
      CHECK(value_type->isIntegerTy(sizeof(T) * 8));
      val = value;
    }
    ir_builder.CreateStore(val, member_ptr);
  } else {
    auto const_default = ll_int<T>(default_value, ctx);
    ir_builder.CreateStore(const_default, member_ptr);
  }
}

std::tuple<llvm::Value*, llvm::Value*> alloc_column(std::string col_name,
                                                    const size_t index,
                                                    const SQLTypeInfo& data_target_info,
                                                    llvm::Value* data_ptr,
                                                    llvm::Value* data_size,
                                                    llvm::Value* data_str_dict_proxy_ptr,
                                                    llvm::LLVMContext& ctx,
                                                    llvm::IRBuilder<>& ir_builder) {
  /*
    Creates a new Column instance of given element type and initialize
    its data ptr and sz members when specified. If data ptr or sz are
    unspecified (have nullptr values) then the corresponding members
    are initialized with NULL and -1, respectively.

    If we are allocating a TextEncodingDict Column type, this function
    adds and populates a int8* pointer to a StringDictProxy object.

    Return a pair of Column allocation (caller should apply
    builder.CreateLoad to it in order to construct a Column instance
    as a value) and a pointer to the Column instance.
   */
  const bool is_text_encoding_dict_type =
      data_target_info.is_string() &&
      data_target_info.get_compression() == kENCODING_DICT;
  llvm::StructType* col_struct_type;
  llvm::Type* data_ptr_llvm_type =
      get_llvm_type_from_sql_column_type(data_target_info, ctx);
  if (is_text_encoding_dict_type) {
    col_struct_type = llvm::StructType::get(
        ctx,
        {
            data_ptr_llvm_type,           /* T* ptr */
            llvm::Type::getInt64Ty(ctx),  /* int64_t sz */
            llvm::Type::getInt8PtrTy(ctx) /* int8_t* string_dictionary_ptr */
        });
  } else {
    std::vector<llvm::Type*> types{
        data_ptr_llvm_type,         /* T* ptr */
        llvm::Type::getInt64Ty(ctx) /* int64_t sz */
    };
    if (data_target_info.is_text_encoding_none()) {
      types.push_back(llvm::Type::getInt8Ty(ctx)); /* int8_t is_null_ */
    }
    col_struct_type = llvm::StructType::get(ctx, types);
  }

  auto col = ir_builder.CreateAlloca(col_struct_type);
  col->setName(col_name);
  auto col_ptr_ptr = ir_builder.CreateStructGEP(col_struct_type, col, 0);
  auto col_sz_ptr = ir_builder.CreateStructGEP(col_struct_type, col, 1);
  auto col_str_dict_ptr = is_text_encoding_dict_type
                              ? ir_builder.CreateStructGEP(col_struct_type, col, 2)
                              : nullptr;
  col_ptr_ptr->setName(col_name + ".ptr");
  col_sz_ptr->setName(col_name + ".sz");
  if (is_text_encoding_dict_type) {
    col_str_dict_ptr->setName(col_name + ".string_dict_proxy");
  }

  initialize_ptr_member(col_ptr_ptr, data_ptr_llvm_type, data_ptr, ir_builder);
  initialize_int_member<int64_t>(col_sz_ptr, data_size, -1, ctx, ir_builder);
  if (data_target_info.is_text_encoding_none()) {
    auto is_null_ptr = ir_builder.CreateStructGEP(col_struct_type, col, 2);
    is_null_ptr->setName(col_name + ".is_null");
    llvm::Value* col_size_lv =
        ir_builder.CreateLoad(col_sz_ptr->getType()->getPointerElementType(), col_sz_ptr);
    llvm::Value* is_null_str_lv = ir_builder.CreateICmpEQ(
        col_size_lv, llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 0));
    auto i8_type = llvm::Type::getInt8Ty(ctx);
    llvm::Value* is_null_i8_lv =
        ir_builder.CreateSelect(is_null_str_lv,
                                llvm::ConstantInt::get(i8_type, 1),
                                llvm::ConstantInt::get(i8_type, 0));
    initialize_int_member<int8_t>(is_null_ptr, is_null_i8_lv, -1, ctx, ir_builder);
  }
  if (is_text_encoding_dict_type) {
    initialize_ptr_member(col_str_dict_ptr,
                          llvm::Type::getInt8PtrTy(ctx),
                          data_str_dict_proxy_ptr,
                          ir_builder);
  }
  auto col_ptr = ir_builder.CreatePointerCast(
      col_ptr_ptr, llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0));
  col_ptr->setName(col_name + "_ptr");
  return {col, col_ptr};
}

llvm::Value* alloc_column_list(std::string col_list_name,
                               const SQLTypeInfo& data_target_info,
                               llvm::Value* data_ptrs,
                               int length,
                               llvm::Value* data_size,
                               llvm::Value* data_str_dict_proxy_ptrs,
                               llvm::LLVMContext& ctx,
                               llvm::IRBuilder<>& ir_builder) {
  /*
    Creates a new ColumnList instance of given element type and initialize
    its members. If data ptr or size are unspecified (have nullptr
    values) then the corresponding members are initialized with NULL
    and -1, respectively.
   */
  llvm::Type* data_ptrs_llvm_type = llvm::Type::getInt8PtrTy(ctx);
  const bool is_text_encoding_dict_type =
      data_target_info.is_string() &&
      data_target_info.get_compression() == kENCODING_DICT;

  llvm::StructType* col_list_struct_type =
      is_text_encoding_dict_type
          ? llvm::StructType::get(
                ctx,
                {
                    data_ptrs_llvm_type,         /* int8_t* ptrs */
                    llvm::Type::getInt64Ty(ctx), /* int64_t length */
                    llvm::Type::getInt64Ty(ctx), /* int64_t size */
                    data_ptrs_llvm_type          /* int8_t* str_dict_proxy_ptrs */
                })
          : llvm::StructType::get(ctx,
                                  {
                                      data_ptrs_llvm_type,         /* int8_t* ptrs */
                                      llvm::Type::getInt64Ty(ctx), /* int64_t length */
                                      llvm::Type::getInt64Ty(ctx)  /* int64_t size */
                                  });

  auto col_list = ir_builder.CreateAlloca(col_list_struct_type);
  col_list->setName(col_list_name);
  auto col_list_ptr_ptr = ir_builder.CreateStructGEP(col_list_struct_type, col_list, 0);
  auto col_list_length_ptr =
      ir_builder.CreateStructGEP(col_list_struct_type, col_list, 1);
  auto col_list_size_ptr = ir_builder.CreateStructGEP(col_list_struct_type, col_list, 2);
  auto col_str_dict_ptr_ptr =
      is_text_encoding_dict_type
          ? ir_builder.CreateStructGEP(col_list_struct_type, col_list, 3)
          : nullptr;

  col_list_ptr_ptr->setName(col_list_name + ".ptrs");
  col_list_length_ptr->setName(col_list_name + ".length");
  col_list_size_ptr->setName(col_list_name + ".size");
  if (is_text_encoding_dict_type) {
    col_str_dict_ptr_ptr->setName(col_list_name + ".string_dict_proxies");
  }

  initialize_ptr_member(col_list_ptr_ptr, data_ptrs_llvm_type, data_ptrs, ir_builder);

  CHECK(length >= 0);
  auto const_length = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), length, true);
  ir_builder.CreateStore(const_length, col_list_length_ptr);

  initialize_int_member<int64_t>(col_list_size_ptr, data_size, -1, ctx, ir_builder);

  if (is_text_encoding_dict_type) {
    initialize_ptr_member(col_str_dict_ptr_ptr,
                          data_str_dict_proxy_ptrs->getType(),
                          data_str_dict_proxy_ptrs,
                          ir_builder);
  }

  auto col_list_ptr = ir_builder.CreatePointerCast(
      col_list_ptr_ptr, llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0));
  col_list_ptr->setName(col_list_name + "_ptrs");
  return col_list_ptr;
}

llvm::Value* alloc_array(std::string arr_name,
                         const size_t index,
                         const SQLTypeInfo& data_target_info,
                         llvm::Value* data_ptr,
                         llvm::Value* data_size,
                         llvm::Value* data_is_null,
                         llvm::LLVMContext& ctx,
                         llvm::IRBuilder<>& ir_builder) {
  /*
    Creates a new Array instance of given element type and initialize
    its data ptr and sz members when specified. If data ptr or sz are
    unspecified (have nullptr values) then the corresponding members
    are initialized with NULL and -1, respectively.

    Return a pointer to Array instance.
   */
  llvm::StructType* arr_struct_type;
  llvm::Type* data_ptr_llvm_type =
      get_llvm_type_from_sql_column_type(data_target_info.get_elem_type(), ctx);
  arr_struct_type =
      llvm::StructType::get(ctx,
                            {
                                data_ptr_llvm_type,          /* T* ptr_ */
                                llvm::Type::getInt64Ty(ctx), /* int64_t size_ */
                                llvm::Type::getInt8Ty(ctx)   /* int8_t is_null_ */
                            });

  auto arr = ir_builder.CreateAlloca(arr_struct_type);
  arr->setName(arr_name);
  auto arr_ptr_ptr = ir_builder.CreateStructGEP(arr_struct_type, arr, 0);
  auto arr_sz_ptr = ir_builder.CreateStructGEP(arr_struct_type, arr, 1);
  auto arr_is_null_ptr = ir_builder.CreateStructGEP(arr_struct_type, arr, 2);
  arr_ptr_ptr->setName(arr_name + ".ptr");
  arr_sz_ptr->setName(arr_name + ".size");
  arr_is_null_ptr->setName(arr_name + ".is_null");

  initialize_ptr_member(arr_ptr_ptr, data_ptr_llvm_type, data_ptr, ir_builder);
  initialize_int_member<int64_t>(arr_sz_ptr, data_size, -1, ctx, ir_builder);
  initialize_int_member<int8_t>(arr_is_null_ptr, data_is_null, -1, ctx, ir_builder);
  auto arr_ptr = ir_builder.CreatePointerCast(
      arr_ptr_ptr, llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0));
  arr_ptr->setName(arr_name + "_pointer");
  return arr_ptr;
}

std::string exprsKey(const std::vector<Analyzer::Expr*>& exprs) {
  std::string result;
  for (const auto& expr : exprs) {
    const auto& ti = expr->get_type_info();
    result += ti.to_string() + ", ";
  }
  return result;
}

}  // namespace

std::shared_ptr<CompilationContext> TableFunctionCompilationContext::compile(
    const TableFunctionExecutionUnit& exe_unit,
    bool emit_only_preflight_fn) {
  auto timer = DEBUG_TIMER(__func__);

  // Here we assume that Executor::tf_code_accessor is cleared when a
  // UDTF implementation is changed. TODO: Ideally, the key should
  // contain a hash of an UDTF implementation string. This could be
  // achieved by including the hash value to the prefix of the UDTF
  // name, for instance.
  CodeCacheKey key{exe_unit.table_func.getName(),
                   exprsKey(exe_unit.input_exprs),
                   exprsKey(exe_unit.target_exprs),
                   std::to_string(emit_only_preflight_fn),
                   std::to_string(co_.device_type == ExecutorDeviceType::GPU)};

  auto cached_code = QueryEngine::getInstance()->tf_code_accessor->get_or_wait(key);
  if (cached_code) {
#ifdef HAVE_CUDA
    if (co_.device_type == ExecutorDeviceType::GPU) {
      auto cached_code_for_gpu =
          std::dynamic_pointer_cast<GpuCompilationContext>(*cached_code);
      CHECK(cached_code_for_gpu);
      CHECK(executor_->data_mgr_);
      CHECK(executor_->data_mgr_->getCudaMgr());
      cached_code_for_gpu->createGpuDeviceCompilationContextForDevices(
          executor_->getAvailableDevicesToProcessQuery(),
          executor_->data_mgr_->getCudaMgr());
      return cached_code_for_gpu;
    }
#endif
    return *cached_code;
  }
  auto compile_start = timer_start();
  auto cgen_state = executor_->getCgenStatePtr();
  CHECK(cgen_state);
  CHECK(cgen_state->module_ == nullptr);
  cgen_state->set_module_shallow_copy(executor_->get_rt_module());

  entry_point_func_ = generate_entry_point(cgen_state);

  generateEntryPoint(exe_unit, emit_only_preflight_fn);

  if (co_.device_type == ExecutorDeviceType::GPU) {
    CHECK(!emit_only_preflight_fn);
    generateGpuKernel();
  }
  std::shared_ptr<CompilationContext> code;
  try {
    code = finalize(emit_only_preflight_fn, compile_start);
  } catch (const std::exception& e) {
    // Erase unsuccesful key and release lock from the get_or_wait(key) call above:
    QueryEngine::getInstance()->tf_code_accessor->erase(key);
    throw;
  }
  // get_or_wait added code with nullptr to cache, here we reset the
  // cached code with the pointer to the generated code:
  QueryEngine::getInstance()->tf_code_accessor->reset(key, code);
  return code;
}

bool TableFunctionCompilationContext::passColumnsByValue(
    const TableFunctionExecutionUnit& exe_unit) {
  bool is_gpu = co_.device_type == ExecutorDeviceType::GPU;
  auto mod = executor_->get_rt_udf_module(is_gpu).get();
  if (mod != nullptr) {
    auto* flag = mod->getModuleFlag("pass_column_arguments_by_value");
    if (auto* cnt = llvm::mdconst::extract_or_null<llvm::ConstantInt>(flag)) {
      return cnt->getZExtValue();
    }
  }

  // fallback to original behavior
  return exe_unit.table_func.isRuntime();
}

void TableFunctionCompilationContext::generateTableFunctionCall(
    const TableFunctionExecutionUnit& exe_unit,
    const std::vector<llvm::Value*>& func_args,
    llvm::BasicBlock* bb_exit,
    llvm::Value* output_row_count_ptr,
    bool emit_only_preflight_fn) {
  auto cgen_state = executor_->getCgenStatePtr();
  // Emit llvm IR code to call the table function
  llvm::LLVMContext& ctx = cgen_state->context_;
  llvm::IRBuilder<>* ir_builder = &cgen_state->ir_builder_;

  std::string func_name =
      (emit_only_preflight_fn ? exe_unit.table_func.getPreFlightFnName()
                              : exe_unit.table_func.getName(false, true));
  llvm::Value* table_func_return =
      cgen_state->emitExternalCall(func_name, get_int_type(32, ctx), func_args);

  table_func_return->setName(emit_only_preflight_fn ? "preflight_check_func_ret"
                                                    : "table_func_ret");

  // If table_func_return is non-negative then store the value in
  // output_row_count and return zero. Otherwise, return
  // table_func_return that negative value contains the error code.
  llvm::BasicBlock* bb_exit_0 =
      llvm::BasicBlock::Create(ctx, ".exit0", entry_point_func_);

  llvm::Constant* const_zero =
      llvm::ConstantInt::get(table_func_return->getType(), 0, true);
  llvm::Value* is_ok = ir_builder->CreateICmpSGE(table_func_return, const_zero);
  ir_builder->CreateCondBr(is_ok, bb_exit_0, bb_exit);

  ir_builder->SetInsertPoint(bb_exit_0);
  llvm::Value* r =
      ir_builder->CreateIntCast(table_func_return, get_int_type(64, ctx), true);
  ir_builder->CreateStore(r, output_row_count_ptr);
  ir_builder->CreateRet(const_zero);

  ir_builder->SetInsertPoint(bb_exit);
  // when table_func_return == TableFunctionErrorCode::NotAnError,
  // then the table function is considered a success while
  // output_row_count_ptr will be uninitialized and the output row
  // count is defined by other means, see QE-877.
  ir_builder->CreateRet(table_func_return);
}

void TableFunctionCompilationContext::generateEntryPoint(
    const TableFunctionExecutionUnit& exe_unit,
    bool emit_only_preflight_fn) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(entry_point_func_);
  CHECK_EQ(entry_point_func_->arg_size(), 7);
  auto arg_it = entry_point_func_->arg_begin();
  const auto mgr_ptr = &*arg_it;
  const auto input_cols_arg = &*(++arg_it);
  const auto input_row_counts_arg = &*(++arg_it);
  const auto input_str_dict_proxies_arg = &*(++arg_it);
  const auto output_buffers_arg = &*(++arg_it);
  const auto output_str_dict_proxies_arg = &*(++arg_it);
  const auto output_row_count_ptr = &*(++arg_it);
  auto cgen_state = executor_->getCgenStatePtr();
  CHECK(cgen_state);
  auto& ctx = cgen_state->context_;

  llvm::BasicBlock* bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", entry_point_func_, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);

  llvm::BasicBlock* bb_exit = llvm::BasicBlock::Create(ctx, ".exit", entry_point_func_);

  llvm::BasicBlock* func_body_bb = llvm::BasicBlock::Create(
      ctx, ".func_body0", cgen_state->ir_builder_.GetInsertBlock()->getParent());

  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  cgen_state->ir_builder_.CreateBr(func_body_bb);

  cgen_state->ir_builder_.SetInsertPoint(func_body_bb);
  auto col_heads = generate_column_heads_load(
      exe_unit.input_exprs.size(), input_cols_arg, cgen_state->ir_builder_, ctx);
  CHECK_EQ(exe_unit.input_exprs.size(), col_heads.size());
  auto row_count_heads = generate_column_heads_load(
      exe_unit.input_exprs.size(), input_row_counts_arg, cgen_state->ir_builder_, ctx);

  auto input_str_dict_proxy_heads = std::vector<llvm::Value*>();
  if (co_.device_type == ExecutorDeviceType::CPU) {
    input_str_dict_proxy_heads = generate_column_heads_load(exe_unit.input_exprs.size(),
                                                            input_str_dict_proxies_arg,
                                                            cgen_state->ir_builder_,
                                                            ctx);
  }
  // The column arguments of C++ UDTFs processed by clang must be
  // passed by reference, see rbc issues 200 and 289.
  auto pass_column_by_value = passColumnsByValue(exe_unit);
  std::vector<llvm::Value*> func_args;
  size_t func_arg_index = 0;
  if (exe_unit.table_func.usesManager()) {
    func_args.push_back(mgr_ptr);
    func_arg_index++;
  }
  int col_index = -1;

  for (size_t i = 0; i < exe_unit.input_exprs.size(); i++) {
    const auto& expr = exe_unit.input_exprs[i];
    const auto& ti = expr->get_type_info();
    if (col_index == -1) {
      func_arg_index += 1;
    }
    if (ti.is_fp()) {
      auto r = cgen_state->ir_builder_.CreateBitCast(
          col_heads[i], get_fp_ptr_type(get_bit_width(ti), ctx));
      llvm::LoadInst* scalar_fp = cgen_state->ir_builder_.CreateLoad(
          r->getType()->getPointerElementType(),
          r,
          "input_scalar_fp." + std::to_string(func_arg_index));
      func_args.push_back(scalar_fp);
      CHECK_EQ(col_index, -1);
    } else if (ti.is_integer() || ti.is_boolean() || ti.is_timestamp() ||
               ti.is_timeinterval()) {
      auto r = cgen_state->ir_builder_.CreateBitCast(
          col_heads[i], get_int_ptr_type(get_bit_width(ti), ctx));
      llvm::LoadInst* scalar_int = cgen_state->ir_builder_.CreateLoad(
          r->getType()->getPointerElementType(),
          r,
          "input_scalar_int." + std::to_string(func_arg_index));
      func_args.push_back(scalar_int);
      CHECK_EQ(col_index, -1);
    } else if (ti.is_text_encoding_none()) {
      auto varchar_size =
          cgen_state->ir_builder_.CreateBitCast(col_heads[i], get_int_ptr_type(64, ctx));
      auto varchar_ptr = cgen_state->ir_builder_.CreateGEP(
          col_heads[i]->getType()->getScalarType()->getPointerElementType(),
          col_heads[i],
          cgen_state->llInt(8));
      auto [varchar_struct, varchar_struct_ptr] = alloc_column(
          std::string("input_varchar_literal.") + std::to_string(func_arg_index),
          i,
          ti,
          varchar_ptr,
          varchar_size,
          nullptr,
          ctx,
          cgen_state->ir_builder_);
      func_args.push_back(
          (pass_column_by_value
               ? cgen_state->ir_builder_.CreateLoad(
                     varchar_struct->getType()->getPointerElementType(), varchar_struct)
               : varchar_struct_ptr));
      CHECK_EQ(col_index, -1);
    } else if (ti.is_column()) {
      auto [col, col_ptr] = alloc_column(
          std::string("input_col.") + std::to_string(func_arg_index),
          i,
          ti.get_elem_type(),
          col_heads[i],
          row_count_heads[i],
          co_.device_type != ExecutorDeviceType::CPU ? nullptr
                                                     : input_str_dict_proxy_heads[i],
          ctx,
          cgen_state->ir_builder_);
      func_args.push_back((pass_column_by_value
                               ? cgen_state->ir_builder_.CreateLoad(
                                     col->getType()->getPointerElementType(), col)
                               : col_ptr));
      CHECK_EQ(col_index, -1);
    } else if (ti.is_column_list()) {
      if (col_index == -1) {
        auto col_list = alloc_column_list(
            std::string("input_col_list.") + std::to_string(func_arg_index),
            ti.get_elem_type(),
            col_heads[i],
            ti.get_dimension(),
            row_count_heads[i],
            input_str_dict_proxy_heads[i],
            ctx,
            cgen_state->ir_builder_);
        func_args.push_back(col_list);
      }
      col_index++;
      if (col_index + 1 == ti.get_dimension()) {
        col_index = -1;
      }
    } else if (ti.is_array()) {
      /*
          Literal array expression is encoded in a contiguous buffer
          with the following memory layout:

          | <array size> | <array is_null> |  <array data>                             |
          |<-- 8 bytes ->|<-- 8 bytes ---->|<-- <array size> * <array element size> -->|

          Notice that while is_null in the `struct Array` has type
          int8_t, in the buffer we use int64_t value to hold the
          is_null state in order to have the array data 64-bit
          aligned.
       */
      auto array_size =
          cgen_state->ir_builder_.CreateBitCast(col_heads[i], get_int_ptr_type(64, ctx));
      auto array_is_null_ptr = cgen_state->ir_builder_.CreateGEP(
          col_heads[i]->getType()->getScalarType()->getPointerElementType(),
          col_heads[i],
          cgen_state->llInt(8));
      auto array_is_null = cgen_state->ir_builder_.CreateLoad(
          array_is_null_ptr->getType()->getPointerElementType(), array_is_null_ptr);

      auto array_ptr = cgen_state->ir_builder_.CreateGEP(
          col_heads[i]->getType()->getScalarType()->getPointerElementType(),
          col_heads[i],
          cgen_state->llInt(16));
      array_size->setName(std::string("array_size.") + std::to_string(func_arg_index));
      array_is_null->setName(std::string("array_is_null.") +
                             std::to_string(func_arg_index));
      array_ptr->setName(std::string("array_ptr.") + std::to_string(func_arg_index));

      auto array_struct_ptr =
          alloc_array(std::string("literal_array.") + std::to_string(func_arg_index),
                      i,
                      ti,
                      array_ptr,
                      array_size,
                      array_is_null,
                      ctx,
                      cgen_state->ir_builder_);

      // passing columns by value is a historical artifact, so no need
      // to support it for array expressions:
      CHECK_EQ(pass_column_by_value, false);
      func_args.push_back(array_struct_ptr);
      CHECK_EQ(col_index, -1);
    } else {
      throw std::runtime_error("Table function input has unsupported type: " +
                               ti.get_type_name());
    }
  }
  auto output_str_dict_proxy_heads =
      co_.device_type == ExecutorDeviceType::CPU
          ? (generate_column_heads_load(exe_unit.target_exprs.size(),
                                        output_str_dict_proxies_arg,
                                        cgen_state->ir_builder_,
                                        ctx))
          : std::vector<llvm::Value*>();

  std::vector<llvm::Value*> output_col_args;
  for (size_t i = 0; i < exe_unit.target_exprs.size(); i++) {
    auto* gep = cgen_state->ir_builder_.CreateGEP(
        output_buffers_arg->getType()->getScalarType()->getPointerElementType(),
        output_buffers_arg,
        cgen_state->llInt(i));
    auto output_load =
        cgen_state->ir_builder_.CreateLoad(gep->getType()->getPointerElementType(), gep);
    const auto& expr = exe_unit.target_exprs[i];
    const auto& ti = expr->get_type_info();
    CHECK(!ti.is_column());       // UDTF output column type is its data type
    CHECK(!ti.is_column_list());  // TODO: when UDTF outputs column_list, convert it to
                                  // output columns
    // UDTF output columns use FlatBuffer storage whenever type supports it
    CHECK_EQ(ti.supportsFlatBuffer(), ti.usesFlatBuffer()) << ti;
    auto [col, col_ptr] = alloc_column(
        std::string("output_col.") + std::to_string(i),
        i,
        ti,
        (co_.device_type == ExecutorDeviceType::GPU
             ? output_load
             : nullptr),  // CPU: set_output_row_size will set the output
                          // Column ptr member
        output_row_count_ptr,
        co_.device_type == ExecutorDeviceType::CPU ? output_str_dict_proxy_heads[i]
                                                   : nullptr,
        ctx,
        cgen_state->ir_builder_);
    if (co_.device_type == ExecutorDeviceType::CPU && !emit_only_preflight_fn) {
      cgen_state->emitExternalCall(
          "TableFunctionManager_register_output_column",
          llvm::Type::getVoidTy(ctx),
          {mgr_ptr, llvm::ConstantInt::get(get_int_type(32, ctx), i, true), col_ptr});
    }
    output_col_args.push_back((pass_column_by_value ? col : col_ptr));
  }

  // output column members must be set before loading column when
  // column instances are passed by value
  if ((exe_unit.table_func.hasOutputSizeKnownPreLaunch() ||
       exe_unit.table_func.hasPreFlightOutputSizer()) &&
      (co_.device_type == ExecutorDeviceType::CPU) && !emit_only_preflight_fn) {
    cgen_state->emitExternalCall(
        "TableFunctionManager_set_output_row_size",
        llvm::Type::getVoidTy(ctx),
        {mgr_ptr,
         cgen_state->ir_builder_.CreateLoad(
             output_row_count_ptr->getType()->getPointerElementType(),
             output_row_count_ptr)});
  }

  if (!emit_only_preflight_fn) {
    for (auto& col : output_col_args) {
      func_args.push_back((pass_column_by_value
                               ? cgen_state->ir_builder_.CreateLoad(
                                     col->getType()->getPointerElementType(), col)
                               : col));
    }
  }

  generateTableFunctionCall(
      exe_unit, func_args, bb_exit, output_row_count_ptr, emit_only_preflight_fn);

  // std::cout << "=================================" << std::endl;
  // entry_point_func_->print(llvm::outs());
  // std::cout << "=================================" << std::endl;

  verify_function_ir(entry_point_func_);
}

void TableFunctionCompilationContext::generateGpuKernel() {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(entry_point_func_);
  std::vector<llvm::Type*> arg_types;
  arg_types.reserve(entry_point_func_->arg_size());
  std::for_each(entry_point_func_->arg_begin(),
                entry_point_func_->arg_end(),
                [&arg_types](const auto& arg) { arg_types.push_back(arg.getType()); });
  CHECK_EQ(arg_types.size(), entry_point_func_->arg_size());

  auto cgen_state = executor_->getCgenStatePtr();
  CHECK(cgen_state);
  auto& ctx = cgen_state->context_;

  std::vector<llvm::Type*> wrapper_arg_types(arg_types.size() + 1);
  wrapper_arg_types[0] = llvm::PointerType::get(get_int_type(32, ctx), 0);
  wrapper_arg_types[1] = arg_types[0];

  for (size_t i = 1; i < arg_types.size(); ++i) {
    wrapper_arg_types[i + 1] = arg_types[i];
  }

  auto wrapper_ft =
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapper_arg_types, false);
  kernel_func_ = llvm::Function::Create(wrapper_ft,
                                        llvm::Function::ExternalLinkage,
                                        "table_func_kernel",
                                        cgen_state->module_);

  auto wrapper_bb_entry = llvm::BasicBlock::Create(ctx, ".entry", kernel_func_, 0);
  llvm::IRBuilder<> b(ctx);
  b.SetInsertPoint(wrapper_bb_entry);
  std::vector<llvm::Value*> loaded_args = {kernel_func_->arg_begin() + 1};
  for (size_t i = 2; i < wrapper_arg_types.size(); ++i) {
    loaded_args.push_back(kernel_func_->arg_begin() + i);
  }
  auto error_lv = b.CreateCall(entry_point_func_, loaded_args);
  b.CreateStore(error_lv, kernel_func_->arg_begin());
  b.CreateRetVoid();
}

std::shared_ptr<CompilationContext> TableFunctionCompilationContext::finalize(
    bool emit_only_preflight_fn,
    std::chrono::steady_clock::time_point& compile_start_timer) {
  auto timer = DEBUG_TIMER(__func__);
  /*
    TODO 1: eliminate need for OverrideFromSrc
    TODO 2: detect and link only the udf's that are needed
  */
  auto cgen_state = executor_->getCgenStatePtr();
  auto is_gpu = co_.device_type == ExecutorDeviceType::GPU;
  if (executor_->has_rt_udf_module(is_gpu)) {
    CodeGenerator::link_udf_module(executor_->get_rt_udf_module(is_gpu),
                                   *(cgen_state->module_),
                                   cgen_state,
                                   llvm::Linker::Flags::OverrideFromSrc);
  }

  LOG(IR) << (emit_only_preflight_fn ? "Pre Flight Function Entry Point IR\n"
                                     : "Table Function Entry Point IR\n")
          << serialize_llvm_object(entry_point_func_);
  VLOG(3) << (emit_only_preflight_fn ? "Pre Flight Function Entry Point IR\n"
                                     : "Table Function Entry Point IR\n")
          << serialize_llvm_object(entry_point_func_);
  std::shared_ptr<CompilationContext> code;
  if (is_gpu) {
    LOG(IR) << "Table Function Kernel IR\n" << serialize_llvm_object(kernel_func_);
    VLOG(3) << "Table Function Kernel IR\n" << serialize_llvm_object(kernel_func_);

    CHECK(executor_);
    executor_->initializeNVPTXBackend();
    CodeGenerator::GPUTarget gpu_target{
        executor_->nvptx_target_machine_.get(), executor_->cudaMgr(), cgen_state, false};
    code = CodeGenerator::generateNativeGPUCode(executor_,
                                                entry_point_func_,
                                                kernel_func_,
                                                {entry_point_func_, kernel_func_},
                                                /*is_gpu_smem_used=*/false,
                                                co_,
                                                gpu_target,
                                                compile_start_timer);
  } else {
    auto ee =
        CodeGenerator::generateNativeCPUCode(entry_point_func_, {entry_point_func_}, co_);
    auto cpu_code = std::make_shared<CpuCompilationContext>(std::move(ee));
    cpu_code->setFunctionPointer(entry_point_func_);
    code = cpu_code;
  }
  LOG(IR) << "End of IR";

  return code;
}
