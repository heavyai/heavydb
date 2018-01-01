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

#include <glog/logging.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

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
  CHECK(false);
  return nullptr;
}

template <class T>
inline llvm::ConstantInt* ll_int(const T v, llvm::LLVMContext& context) {
  return static_cast<llvm::ConstantInt*>(llvm::ConstantInt::get(get_int_type(sizeof(v) * 8, context), v));
}

inline llvm::ConstantInt* ll_bool(const bool v, llvm::LLVMContext& context) {
  return static_cast<llvm::ConstantInt*>(llvm::ConstantInt::get(get_int_type(1, context), v));
}
