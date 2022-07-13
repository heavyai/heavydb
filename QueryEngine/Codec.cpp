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

#include "Codec.h"
#include "LLVMGlobalContext.h"
#include "Logger/Logger.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

FixedWidthInt::FixedWidthInt(const size_t byte_width) : byte_width_{byte_width} {}

llvm::Instruction* FixedWidthInt::codegenDecode(llvm::Value* byte_stream,
                                                llvm::Value* pos,
                                                llvm::Module* llvm_module) const {
  auto& context = llvm_module->getContext();
  auto f = llvm_module->getFunction("fixed_width_int_decode");
  CHECK(f);
  llvm::Value* args[] = {
      byte_stream,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), byte_width_),
      pos};
  return llvm::CallInst::Create(f, args);
}

FixedWidthUnsigned::FixedWidthUnsigned(const size_t byte_width)
    : byte_width_{byte_width} {}

llvm::Instruction* FixedWidthUnsigned::codegenDecode(llvm::Value* byte_stream,
                                                     llvm::Value* pos,
                                                     llvm::Module* llvm_module) const {
  auto& context = llvm_module->getContext();
  auto f = llvm_module->getFunction("fixed_width_unsigned_decode");
  CHECK(f);
  llvm::Value* args[] = {
      byte_stream,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), byte_width_),
      pos};
  return llvm::CallInst::Create(f, args);
}

DiffFixedWidthInt::DiffFixedWidthInt(const size_t byte_width, const int64_t baseline)
    : byte_width_{byte_width}, baseline_{baseline} {}

llvm::Instruction* DiffFixedWidthInt::codegenDecode(llvm::Value* byte_stream,
                                                    llvm::Value* pos,
                                                    llvm::Module* llvm_module) const {
  auto& context = llvm_module->getContext();
  auto f = llvm_module->getFunction("diff_fixed_width_int_decode");
  CHECK(f);
  llvm::Value* args[] = {
      byte_stream,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), byte_width_),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), baseline_),
      pos};
  return llvm::CallInst::Create(f, args);
}

FixedWidthReal::FixedWidthReal(const bool is_double) : is_double_(is_double) {}

llvm::Instruction* FixedWidthReal::codegenDecode(llvm::Value* byte_stream,
                                                 llvm::Value* pos,
                                                 llvm::Module* llvm_module) const {
  auto f = llvm_module->getFunction(is_double_ ? "fixed_width_double_decode"
                                               : "fixed_width_float_decode");
  CHECK(f);
  llvm::Value* args[] = {byte_stream, pos};
  return llvm::CallInst::Create(f, args);
}

FixedWidthSmallDate::FixedWidthSmallDate(const size_t byte_width)
    : byte_width_{byte_width}, null_val_{byte_width == 4 ? NULL_INT : NULL_SMALLINT} {}

llvm::Instruction* FixedWidthSmallDate::codegenDecode(llvm::Value* byte_stream,
                                                      llvm::Value* pos,
                                                      llvm::Module* llvm_module) const {
  auto& context = llvm_module->getContext();
  auto f = llvm_module->getFunction("fixed_width_small_date_decode");
  CHECK(f);
  llvm::Value* args[] = {
      byte_stream,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), byte_width_),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), null_val_),
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), ret_null_val_),
      pos};
  return llvm::CallInst::Create(f, args);
}
