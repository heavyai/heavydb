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

#pragma once

#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/raw_os_ostream.h>

#include "Shared/Logger.h"

#if LLVM_VERSION_MAJOR >= 10
#define LLVM_ALIGN(alignment) llvm::Align(alignment)
#define LLVM_MAYBE_ALIGN(alignment) llvm::MaybeAlign(alignment)
#else
#define LLVM_ALIGN(alignment) alignment
#define LLVM_MAYBE_ALIGN(alignment) alignment
#endif

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
      return llvm::VectorType::get(llvm::Type::getInt64Ty(context), count);
    case 32:
      return llvm::VectorType::get(llvm::Type::getInt32Ty(context), count);
      break;
    case 16:
      return llvm::VectorType::get(llvm::Type::getInt16Ty(context), count);
      break;
    case 8:
      return llvm::VectorType::get(llvm::Type::getInt8Ty(context), count);
      break;
    case 1:
      return llvm::VectorType::get(llvm::Type::getInt1Ty(context), count);
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

template <class T>
inline llvm::ConstantInt* ll_int(const T v, llvm::LLVMContext& context) {
  return static_cast<llvm::ConstantInt*>(
      llvm::ConstantInt::get(get_int_type(sizeof(v) * 8, context), v));
}

inline llvm::ConstantInt* ll_bool(const bool v, llvm::LLVMContext& context) {
  return static_cast<llvm::ConstantInt*>(
      llvm::ConstantInt::get(get_int_type(1, context), v));
}

llvm::Module* read_template_module(llvm::LLVMContext& context);

template <class T>
std::string serialize_llvm_object(const T* llvm_obj) {
  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  llvm_obj->print(os);
  os.flush();
  return ss.str();
}

void verify_function_ir(const llvm::Function* func);
