/*
 * Copyright 2019 OmniSci, Inc.
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

#include "ResultSetReductionCodegen.h"

#include "IRCodegenUtils.h"
#include "LoopControlFlow/JoinLoop.h"
#include "ResultSetReductionJIT.h"
#include "ResultSetReductionOps.h"

#include <llvm/IR/Instructions.h>

llvm::Type* llvm_type(const Type type, llvm::LLVMContext& ctx) {
  switch (type) {
    case Type::Int1: {
      return get_int_type(1, ctx);
    }
    case Type::Int8: {
      return get_int_type(8, ctx);
    }
    case Type::Int32: {
      return get_int_type(32, ctx);
    }
    case Type::Int64: {
      return get_int_type(64, ctx);
    }
    case Type::Float: {
      return get_fp_type(32, ctx);
    }
    case Type::Double: {
      return get_fp_type(64, ctx);
    }
    case Type::Void: {
      return llvm::Type::getVoidTy(ctx);
    }
    case Type::Int8Ptr: {
      return llvm::PointerType::get(get_int_type(8, ctx), 0);
    }
    case Type::Int32Ptr: {
      return llvm::PointerType::get(get_int_type(32, ctx), 0);
    }
    case Type::Int64Ptr: {
      return llvm::PointerType::get(get_int_type(64, ctx), 0);
    }
    case Type::FloatPtr: {
      return llvm::Type::getFloatPtrTy(ctx);
    }
    case Type::DoublePtr: {
      return llvm::Type::getDoublePtrTy(ctx);
    }
    case Type::VoidPtr: {
      return llvm::PointerType::get(get_int_type(8, ctx), 0);
    }
    case Type::Int64PtrPtr: {
      return llvm::PointerType::get(llvm::PointerType::get(get_int_type(64, ctx), 0), 0);
    }
    default: {
      LOG(FATAL) << "Argument type not supported: " << static_cast<int>(type);
      break;
    }
  }
  UNREACHABLE();
  return nullptr;
}

namespace {

// Convert an IR predicate to the corresponding LLVM one.
llvm::ICmpInst::Predicate llvm_predicate(const ICmp::Predicate predicate) {
  switch (predicate) {
    case ICmp::Predicate::EQ: {
      return llvm::ICmpInst::ICMP_EQ;
    }
    case ICmp::Predicate::NE: {
      return llvm::ICmpInst::ICMP_NE;
    }
    default: {
      LOG(FATAL) << "Invalid predicate: " << static_cast<int>(predicate);
    }
  }
  UNREACHABLE();
  return llvm::ICmpInst::ICMP_EQ;
}

// Convert an IR binary operator type to the corresponding LLVM one.
llvm::BinaryOperator::BinaryOps llvm_binary_op(const BinaryOperator::BinaryOp op) {
  switch (op) {
    case BinaryOperator::BinaryOp::Add: {
      return llvm::Instruction::Add;
    }
    case BinaryOperator::BinaryOp::Mul: {
      return llvm::Instruction::Mul;
    }
    default: {
      LOG(FATAL) << "Invalid binary operator: " << static_cast<int>(op);
    }
  }
  UNREACHABLE();
  return llvm::Instruction::Add;
}

// Convert an IR cast operator type to the corresponding LLVM one.
llvm::Instruction::CastOps llvm_cast_op(const Cast::CastOp op) {
  switch (op) {
    case Cast::CastOp::Trunc: {
      return llvm::Instruction::Trunc;
    }
    case Cast::CastOp::SExt: {
      return llvm::Instruction::SExt;
    }
    case Cast::CastOp::BitCast: {
      return llvm::Instruction::BitCast;
    }
    default: {
      LOG(FATAL) << "Invalid cast operator: " << static_cast<int>(op);
    }
  }
  UNREACHABLE();
  return llvm::Instruction::SExt;
}

// Emit an early return from a function when the provided 'cond' is true, which the caller
// code can use when entries are empty or the watchdog is triggered. For functions which
// return void, the specified error code is ignored. For functions which return an
// integer, the error code is returned.
void return_early(llvm::Value* cond,
                  const ReductionCode& reduction_code,
                  llvm::Function* func,
                  llvm::Value* error_code) {
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto early_return = llvm::BasicBlock::Create(ctx, ".early_return", func, 0);
  const auto do_reduction = llvm::BasicBlock::Create(ctx, ".do_reduction", func, 0);
  cgen_state->ir_builder_.CreateCondBr(cond, early_return, do_reduction);
  cgen_state->ir_builder_.SetInsertPoint(early_return);

  if (func->getReturnType()->isVoidTy()) {
    cgen_state->ir_builder_.CreateRetVoid();
  } else {
    CHECK(error_code);
    cgen_state->ir_builder_.CreateRet(error_code);
  }

  cgen_state->ir_builder_.SetInsertPoint(do_reduction);
}

// Returns the corresponding LLVM value for the given IR value.
llvm::Value* mapped_value(const Value* val,
                          const std::unordered_map<const Value*, llvm::Value*>& m) {
  if (val) {
    const auto it = m.find(val);
    CHECK(it != m.end());
    return it->second;
  } else {
    return nullptr;
  }
}

// Returns the corresponding LLVM function for the given IR function.
llvm::Function* mapped_function(
    const Function* function,
    const std::unordered_map<const Function*, llvm::Function*>& f) {
  const auto it = f.find(function);
  CHECK(it != f.end());
  return it->second;
}

// Given a list of IR values and the mapping, return the list of corresponding LLVM IR
// values.
std::vector<llvm::Value*> llvm_args(
    const std::vector<const Value*> args,
    const std::unordered_map<const Value*, llvm::Value*>& m) {
  std::vector<llvm::Value*> llvm_args;
  std::transform(
      args.begin(), args.end(), std::back_inserter(llvm_args), [&m](const Value* value) {
        return mapped_value(value, m);
      });
  return llvm_args;
}

void translate_for(const For* for_loop,
                   Function* ir_reduce_loop,
                   const ReductionCode& reduction_code,
                   std::unordered_map<const Value*, llvm::Value*>& m,
                   const std::unordered_map<const Function*, llvm::Function*>& f);

// Translate a list of instructions to LLVM IR.
void translate_body(const std::vector<std::unique_ptr<Instruction>>& body,
                    const Function* function,
                    llvm::Function* llvm_function,
                    const ReductionCode& reduction_code,
                    std::unordered_map<const Value*, llvm::Value*>& m,
                    const std::unordered_map<const Function*, llvm::Function*>& f) {
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  for (const auto& instr : body) {
    const auto instr_ptr = instr.get();
    llvm::Value* translated{nullptr};
    if (auto gep = dynamic_cast<const GetElementPtr*>(instr_ptr)) {
      translated = cgen_state->ir_builder_.CreateGEP(
          mapped_value(gep->base(), m), mapped_value(gep->index(), m), gep->label());
    } else if (auto load = dynamic_cast<const Load*>(instr_ptr)) {
      translated = cgen_state->ir_builder_.CreateLoad(mapped_value(load->source(), m),
                                                      load->label());
    } else if (auto icmp = dynamic_cast<const ICmp*>(instr_ptr)) {
      translated = cgen_state->ir_builder_.CreateICmp(llvm_predicate(icmp->predicate()),
                                                      mapped_value(icmp->lhs(), m),
                                                      mapped_value(icmp->rhs(), m),
                                                      icmp->label());
    } else if (auto binary_operator = dynamic_cast<const BinaryOperator*>(instr_ptr)) {
      translated =
          cgen_state->ir_builder_.CreateBinOp(llvm_binary_op(binary_operator->op()),
                                              mapped_value(binary_operator->lhs(), m),
                                              mapped_value(binary_operator->rhs(), m),
                                              binary_operator->label());
    } else if (auto cast = dynamic_cast<const Cast*>(instr_ptr)) {
      translated = cgen_state->ir_builder_.CreateCast(llvm_cast_op(cast->op()),
                                                      mapped_value(cast->source(), m),
                                                      llvm_type(cast->type(), ctx),
                                                      cast->label());
    } else if (auto ret = dynamic_cast<const Ret*>(instr_ptr)) {
      if (ret->value()) {
        cgen_state->ir_builder_.CreateRet(mapped_value(ret->value(), m));
      } else {
        cgen_state->ir_builder_.CreateRetVoid();
      }
    } else if (auto call = dynamic_cast<const Call*>(instr_ptr)) {
      std::vector<llvm::Value*> llvm_args;
      const auto args = call->arguments();
      std::transform(args.begin(),
                     args.end(),
                     std::back_inserter(llvm_args),
                     [&m](const Value* value) { return mapped_value(value, m); });
      if (call->callee()) {
        translated = cgen_state->ir_builder_.CreateCall(
            mapped_function(call->callee(), f), llvm_args, call->label());
      } else {
        translated = cgen_state->emitCall(call->callee_name(), llvm_args);
      }
    } else if (auto external_call = dynamic_cast<const ExternalCall*>(instr_ptr)) {
      translated = cgen_state->emitExternalCall(external_call->callee_name(),
                                                llvm_type(external_call->type(), ctx),
                                                llvm_args(external_call->arguments(), m));
    } else if (auto alloca = dynamic_cast<const Alloca*>(instr_ptr)) {
      translated = cgen_state->ir_builder_.CreateAlloca(
          llvm_type(pointee_type(alloca->type()), ctx),
          mapped_value(alloca->array_size(), m),
          alloca->label());
    } else if (auto memcpy = dynamic_cast<const MemCpy*>(instr_ptr)) {
      cgen_state->ir_builder_.CreateMemCpy(mapped_value(memcpy->dest(), m),
                                           0,
                                           mapped_value(memcpy->source(), m),
                                           0,
                                           mapped_value(memcpy->size(), m));
    } else if (auto ret_early = dynamic_cast<const ReturnEarly*>(instr_ptr)) {
      return_early(mapped_value(ret_early->cond(), m),
                   reduction_code,
                   llvm_function,
                   mapped_value(ret_early->error_code(), m));
    } else if (auto for_loop = dynamic_cast<const For*>(instr_ptr)) {
      translate_for(for_loop, reduction_code.ir_reduce_loop.get(), reduction_code, m, f);
    } else {
      LOG(FATAL) << "Instruction not supported yet";
    }
    if (translated) {
      const auto it_ok = m.emplace(instr_ptr, translated);
      CHECK(it_ok.second);
    }
  }
}

// Translate a loop to LLVM IR, using existing loop construction facilities.
void translate_for(const For* for_loop,
                   Function* ir_reduce_loop,
                   const ReductionCode& reduction_code,
                   std::unordered_map<const Value*, llvm::Value*>& m,
                   const std::unordered_map<const Function*, llvm::Function*>& f) {
  auto cgen_state = reduction_code.cgen_state.get();
  const auto bb_entry = cgen_state->ir_builder_.GetInsertBlock();
  auto& ctx = cgen_state->context_;
  const auto i64_type = get_int_type(64, cgen_state->context_);
  const auto end_index = mapped_value(for_loop->end(), m);
  const auto start_index = mapped_value(for_loop->start(), m);
  // The start and end indices are absolute. Subtract the start index from the iterator.
  const auto iteration_count =
      cgen_state->ir_builder_.CreateSub(end_index, start_index, "iteration_count");
  const auto upper_bound = cgen_state->ir_builder_.CreateSExt(iteration_count, i64_type);
  const auto bb_exit =
      llvm::BasicBlock::Create(ctx, ".exit", mapped_function(ir_reduce_loop, f));
  JoinLoop join_loop(
      JoinLoopKind::UpperBound,
      JoinType::INNER,
      [upper_bound](const std::vector<llvm::Value*>& v) {
        JoinLoopDomain domain{{0}};
        domain.upper_bound = upper_bound;
        return domain;
      },
      nullptr,
      nullptr,
      nullptr,
      "reduction_loop");
  const auto bb_loop_body = JoinLoop::codegen(
      {join_loop},
      [cgen_state, for_loop, ir_reduce_loop, &f, &m, &reduction_code](
          const std::vector<llvm::Value*>& iterators) {
        const auto loop_body_bb = llvm::BasicBlock::Create(
            cgen_state->context_,
            ".loop_body",
            cgen_state->ir_builder_.GetInsertBlock()->getParent());
        cgen_state->ir_builder_.SetInsertPoint(loop_body_bb);
        // Make the iterator the same type as start and end indices (32-bit integer).
        const auto loop_iter =
            cgen_state->ir_builder_.CreateTrunc(iterators.back(),
                                                get_int_type(32, cgen_state->context_),
                                                "relative_entry_idx");
        m.emplace(for_loop->iter(), loop_iter);
        translate_body(for_loop->body(),
                       ir_reduce_loop,
                       mapped_function(ir_reduce_loop, f),
                       reduction_code,
                       m,
                       f);
        return loop_body_bb;
      },
      nullptr,
      bb_exit,
      cgen_state->ir_builder_);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  cgen_state->ir_builder_.CreateBr(bb_loop_body);
  cgen_state->ir_builder_.SetInsertPoint(bb_exit);
}

// Create the entry basic block into an initially empty function.
void create_entry_block(llvm::Function* function, CgenState* cgen_state) {
  const auto bb_entry =
      llvm::BasicBlock::Create(cgen_state->context_, ".entry", function, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
}

}  // namespace

void translate_function(const Function* function,
                        llvm::Function* llvm_function,
                        const ReductionCode& reduction_code,
                        const std::unordered_map<const Function*, llvm::Function*>& f) {
  auto cgen_state = reduction_code.cgen_state.get();
  create_entry_block(llvm_function, cgen_state);
  // Set the value mapping based on the input arguments.
  std::unordered_map<const Value*, llvm::Value*> m;
  auto llvm_arg_it = llvm_function->arg_begin();
  for (size_t arg_idx = 0; arg_idx < function->arg_types().size(); ++arg_idx) {
    llvm::Value* llvm_arg = &(*llvm_arg_it);
    const auto it_ok = m.emplace(function->arg(arg_idx), llvm_arg);
    CHECK(it_ok.second);
    ++llvm_arg_it;
  }
  // Add mapping for the constants used by the function.
  for (const auto& constant : function->constants()) {
    llvm::Value* constant_llvm{nullptr};
    switch (constant->type()) {
      case Type::Int8: {
        constant_llvm =
            cgen_state->llInt<int8_t>(static_cast<ConstantInt*>(constant.get())->value());
        break;
      }
      case Type::Int32: {
        constant_llvm = cgen_state->llInt<int32_t>(
            static_cast<ConstantInt*>(constant.get())->value());
        break;
      }
      case Type::Int64: {
        constant_llvm = cgen_state->llInt<int64_t>(
            static_cast<ConstantInt*>(constant.get())->value());
        break;
      }
      case Type::Float: {
        constant_llvm = cgen_state->llFp(
            static_cast<float>(static_cast<ConstantFP*>(constant.get())->value()));
        break;
      }
      case Type::Double: {
        constant_llvm =
            cgen_state->llFp(static_cast<ConstantFP*>(constant.get())->value());
        break;
      }
      default: {
        LOG(FATAL) << "Constant type not supported: "
                   << static_cast<int>(constant->type());
      }
    }
    CHECK(constant_llvm);
    const auto it_ok = m.emplace(constant.get(), constant_llvm);
    CHECK(it_ok.second);
  }
  translate_body(function->body(), function, llvm_function, reduction_code, m, f);
  verify_function_ir(llvm_function);
}
