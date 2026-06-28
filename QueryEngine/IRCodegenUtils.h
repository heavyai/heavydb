/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/ADT/Twine.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_os_ostream.h>

#include "Logger/Logger.h"
#include "Shared/sqltypes.h"

#define LLVM_ALIGN(alignment) llvm::Align(alignment)
#define LLVM_MAYBE_ALIGN(alignment) llvm::MaybeAlign(alignment)

inline llvm::ArrayType* get_int_array_type(int const width,
                                           int count,
                                           llvm::LLVMContext& context) {
  switch (width) {
    case 64:
      return llvm::ArrayType::get(llvm::Type::getInt64Ty(context), count);
    case 32:
      return llvm::ArrayType::get(llvm::Type::getInt32Ty(context), count);
      break;
    case 16:
      return llvm::ArrayType::get(llvm::Type::getInt16Ty(context), count);
      break;
    case 8:
      return llvm::ArrayType::get(llvm::Type::getInt8Ty(context), count);
      break;
    case 1:
      return llvm::ArrayType::get(llvm::Type::getInt1Ty(context), count);
      break;
    default:
      LOG(FATAL) << "Unsupported integer width: " << width;
  }
  return nullptr;
}

inline llvm::VectorType* get_int_vector_type(int const width,
                                             int count,
                                             llvm::LLVMContext& context) {
  switch (width) {
    case 64:
      return llvm::VectorType::get(llvm::Type::getInt64Ty(context), count, false);
    case 32:
      return llvm::VectorType::get(llvm::Type::getInt32Ty(context), count, false);
      break;
    case 16:
      return llvm::VectorType::get(llvm::Type::getInt16Ty(context), count, false);
      break;
    case 8:
      return llvm::VectorType::get(llvm::Type::getInt8Ty(context), count, false);
      break;
    case 1:
      return llvm::VectorType::get(llvm::Type::getInt1Ty(context), count, false);
      break;
    default:
      LOG(FATAL) << "Unsupported integer width: " << width;
  }
  return nullptr;
}

inline llvm::Type* get_int_type(const int width, llvm::LLVMContext& context) {
  switch (width) {
    case 64:
      return llvm::Type::getInt64Ty(context);
    case 32:
      return llvm::Type::getInt32Ty(context);
      break;
    case 16:
      return llvm::Type::getInt16Ty(context);
      break;
    case 8:
      return llvm::Type::getInt8Ty(context);
      break;
    case 1:
      return llvm::Type::getInt1Ty(context);
      break;
    default:
      LOG(FATAL) << "Unsupported integer width: " << width;
  }
  UNREACHABLE();
  return nullptr;
}

inline llvm::Type* get_fp_type(const int width, llvm::LLVMContext& context) {
  switch (width) {
    case 64:
      return llvm::Type::getDoubleTy(context);
    case 32:
      return llvm::Type::getFloatTy(context);
    default:
      LOG(FATAL) << "Unsupported floating point width: " << width;
  }
  UNREACHABLE();
  return nullptr;
}

// Opaque pointer type (LLVM 15+).
//
// Under LLVM 15+ opaque pointers, every GEP and load must specify element type
// explicitly. For [N x T] allocas, either use typed_array_element_ptr (two-index
// {0, idx} GEP with ArrayType), or typed_alloca_element_ptr (bitcast to T* and
// single-index GEP). Never use single-index GEP with ArrayType as the pointee type.
//
// Never CreatePointerCast(ptr, integer ty); use CgenState::castToTypeIn (PtrToInt +
// Trunc). Pointer-to-pointer bitcasts (including address-space changes) remain valid.
inline llvm::PointerType* opaque_ptr_ty(llvm::LLVMContext& ctx, unsigned addr_space = 0) {
  return llvm::PointerType::get(ctx, addr_space);
}

// Pointer type for function signatures and typed GEP/load (opaque on LLVM 15+).
inline llvm::PointerType* typed_ptr_ty(llvm::Type* pointee_ty, unsigned addr_space = 0) {
  (void)pointee_ty;
  return llvm::PointerType::get(pointee_ty->getContext(), addr_space);
}

inline llvm::PointerType* get_fp_ptr_type(const int width, llvm::LLVMContext& context) {
  return typed_ptr_ty(get_fp_type(width, context), 0);
}

inline llvm::PointerType* get_int_ptr_type(const int width, llvm::LLVMContext& context) {
  return typed_ptr_ty(get_int_type(width, context), 0);
}

inline llvm::Value* typed_gep(llvm::IRBuilder<>& b,
                              llvm::Type* pointee_ty,
                              llvm::Value* ptr,
                              llvm::Value* idx) {
  return b.CreateGEP(pointee_ty, ptr, idx);
}

// Element idx of an alloca'd [N x ElemTy] array (steps by sizeof(ElemTy)).
// Equivalent to master-era GEP(alloca->getPointerElementType(), alloca, idx).
inline llvm::Value* typed_alloca_element_ptr(llvm::IRBuilder<>& b,
                                             llvm::Type* elem_ty,
                                             llvm::Value* alloca_ptr,
                                             llvm::Value* idx) {
  auto* const elem_ptr = b.CreateBitCast(alloca_ptr, typed_ptr_ty(elem_ty, 0));
  return typed_gep(b, elem_ty, elem_ptr, idx);
}

// Element idx of a [N x ElemTy] array via two-index {0, idx} GEP.
inline llvm::Value* typed_array_element_ptr(llvm::IRBuilder<>& b,
                                            llvm::ArrayType* arr_ty,
                                            llvm::Value* arr_ptr,
                                            llvm::Value* idx) {
  auto& ctx = b.getContext();
  return b.CreateGEP(
      arr_ty, arr_ptr, {llvm::ConstantInt::get(get_int_type(32, ctx), 0), idx});
}

inline llvm::Value* typed_load(llvm::IRBuilder<>& b,
                               llvm::Type* load_ty,
                               llvm::Value* ptr,
                               const llvm::Twine& name = "") {
  return b.CreateLoad(load_ty, ptr, name);
}

// Window COUNT state uses i32* for float aggregate args (see agg_count_float) and
// i64* otherwise. Loading with the wrong width reads past the slot and corrupts the
// result (WindowFunctionAggregate / ExecuteTest).
inline llvm::Value* typed_aggregate_count_load(llvm::IRBuilder<>& b,
                                               llvm::LLVMContext& ctx,
                                               llvm::Value* state_ptr,
                                               const SQLTypeInfo& window_func_ti) {
  if (window_func_ti.get_type() == kFLOAT) {
    return typed_load(b, get_int_type(32, ctx), state_ptr);
  }
  return typed_load(b, get_int_type(64, ctx), state_ptr);
}

template <class T>
inline llvm::ConstantInt* ll_int(const T v, llvm::LLVMContext& context) {
  return static_cast<llvm::ConstantInt*>(
      llvm::ConstantInt::get(get_int_type(sizeof(v) * 8, context), v));
}

inline llvm::ConstantInt* ll_bool(const bool v, llvm::LLVMContext& context) {
  return static_cast<llvm::ConstantInt*>(
      llvm::ConstantInt::get(get_int_type(1, context), v));
}

template <class T>
std::string serialize_llvm_object(const T* llvm_obj) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << *llvm_obj;
  os.flush();
  return str;
}

void verify_function_ir(const llvm::Function* func);
