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

#include "ResultSetReductionInterpreter.h"
#include "ResultSetReductionInterpreterStubs.h"

thread_local size_t g_value_id;

namespace {

// Extract types of the given values.
std::vector<Type> get_value_types(const std::vector<const Value*>& values) {
  std::vector<Type> value_types;
  value_types.reserve(value_types.size());
  std::transform(values.begin(),
                 values.end(),
                 std::back_inserter(value_types),
                 [](const Value* value) { return value->type(); });
  return value_types;
}

// For an alloca buffer, return the element size.
size_t get_element_size(const Type element_type) {
  switch (element_type) {
    case Type::Int8Ptr: {
      return sizeof(int8_t);
    }
    case Type::Int64PtrPtr: {
      return sizeof(int64_t*);
    }
    default: {
      LOG(FATAL) << "Base pointer type not supported: " << static_cast<int>(element_type);
      break;
    }
  }
  return 0;
}

}  // namespace

// Implements execution for all the operators. Caller is responsible for stopping
// evaluation when the return value is set.
class ReductionInterpreterImpl {
 public:
  ReductionInterpreterImpl(const size_t executor_id,
                           const std::vector<ReductionInterpreter::EvalValue>& vars)
      : executor_id_(executor_id), vars_(vars) {}

  std::optional<ReductionInterpreter::EvalValue> ret() const { return ret_; }

 public:
  size_t getExecutorId() const { return executor_id_; }
  static void runGetElementPtr(const Instruction* instruction,
                               ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto gep = static_cast<const GetElementPtr*>(instruction);
    const auto element_size = get_element_size(gep->base()->type());
    const auto base = interpreter->vars_[gep->base()->id()];
    const auto index = interpreter->vars_[gep->index()->id()];
    auto result_ptr =
        reinterpret_cast<const int8_t*>(base.ptr) + index.int_val * element_size;
    interpreter->setVar(gep, ReductionInterpreter::MakeEvalValue(result_ptr));
  }

  static void runLoad(const Instruction* instruction,
                      ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto load = static_cast<const Load*>(instruction);
    const auto source_type = load->source()->type();
    CHECK(is_pointer_type(source_type));
    const auto source = interpreter->vars_[load->source()->id()];
    switch (source_type) {
      case Type::Int8Ptr: {
        const auto int_val = *reinterpret_cast<const int8_t*>(source.ptr);
        interpreter->setVar(load, ReductionInterpreter::MakeEvalValue(int_val));
        break;
      }
      case Type::Int32Ptr: {
        const auto int_val = *reinterpret_cast<const int32_t*>(source.ptr);
        interpreter->setVar(load, ReductionInterpreter::MakeEvalValue(int_val));
        break;
      }
      case Type::Int64Ptr: {
        const auto int_val = *reinterpret_cast<const int64_t*>(source.ptr);
        interpreter->setVar(load, ReductionInterpreter::MakeEvalValue(int_val));
        break;
      }
      case Type::FloatPtr: {
        const auto float_val = *reinterpret_cast<const float*>(source.ptr);
        interpreter->setVar(load, ReductionInterpreter::MakeEvalValue(float_val));
        break;
      }
      case Type::DoublePtr: {
        const auto double_val = *reinterpret_cast<const double*>(source.ptr);
        interpreter->setVar(load, ReductionInterpreter::MakeEvalValue(double_val));
        break;
      }
      case Type::Int64PtrPtr: {
        const auto int_ptr_val = *reinterpret_cast<const int64_t* const*>(source.ptr);
        interpreter->setVar(load, ReductionInterpreter::MakeEvalValue(int_ptr_val));
        break;
      }
      default: {
        LOG(FATAL) << "Source pointer type not supported: "
                   << static_cast<int>(source_type);
      }
    }
  }

  static void runICmp(const Instruction* instruction,
                      ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto icmp = static_cast<const ICmp*>(instruction);
    CHECK(is_integer_type(icmp->lhs()->type()));
    CHECK(is_integer_type(icmp->rhs()->type()));
    const auto lhs = interpreter->vars_[icmp->lhs()->id()];
    const auto rhs = interpreter->vars_[icmp->rhs()->id()];
    bool result = false;
    switch (icmp->predicate()) {
      case ICmp::Predicate::EQ: {
        result = lhs.int_val == rhs.int_val;
        break;
      }
      case ICmp::Predicate::NE: {
        result = lhs.int_val != rhs.int_val;
        break;
      }
      default: {
        LOG(FATAL) << "Predicate not supported: " << static_cast<int>(icmp->predicate());
      }
    }
    interpreter->setVar(icmp, ReductionInterpreter::MakeEvalValue(result));
  }

  static void runBinaryOperator(const Instruction* instruction,
                                ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto binary_operator = static_cast<const BinaryOperator*>(instruction);
    CHECK(is_integer_type(binary_operator->type()));
    const auto lhs = interpreter->vars_[binary_operator->lhs()->id()];
    const auto rhs = interpreter->vars_[binary_operator->rhs()->id()];
    int64_t result = 0;
    switch (binary_operator->op()) {
      case BinaryOperator::BinaryOp::Add: {
        result = lhs.int_val + rhs.int_val;
        break;
      }
      case BinaryOperator::BinaryOp::Mul: {
        result = lhs.int_val * rhs.int_val;
        break;
      }
      default: {
        LOG(FATAL) << "Binary operator not supported: "
                   << static_cast<int>(binary_operator->op());
      }
    }
    interpreter->setVar(binary_operator, ReductionInterpreter::MakeEvalValue(result));
  }

  static void runCast(const Instruction* instruction,
                      ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto cast = static_cast<const Cast*>(instruction);
    const auto source = interpreter->vars_[cast->source()->id()];
    // Given that evaluated values store all values as int64_t or void*, Trunc and SExt
    // are no-op. The information about the type is already part of the destination.
    switch (cast->op()) {
      case Cast::CastOp::Trunc:
      case Cast::CastOp::SExt: {
        CHECK(is_integer_type(cast->source()->type()));
        interpreter->setVar(cast, ReductionInterpreter::MakeEvalValue(source.int_val));
        break;
      }
      case Cast::CastOp::BitCast: {
        CHECK(is_pointer_type(cast->source()->type()));
        interpreter->setVar(cast, ReductionInterpreter::MakeEvalValue(source.ptr));
        break;
      }
      default: {
        LOG(FATAL) << "Cast operator not supported: " << static_cast<int>(cast->op());
      }
    }
  }

  static void runRet(const Instruction* instruction,
                     ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto ret = static_cast<const Ret*>(instruction);
    if (ret->type() == Type::Void) {
      // Even if the returned type is void, the return value still needs to be set to
      // something to inform the caller that it should stop evaluating.
      interpreter->ret_ = ReductionInterpreter::EvalValue{};
    } else {
      interpreter->ret_ = interpreter->vars_[ret->value()->id()];
    }
  }

  static void runCall(const Instruction* instruction,
                      ReductionInterpreterImpl* interpreter) {
    auto executor_id = interpreter->getExecutorId();
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto call = static_cast<const Call*>(instruction);
    if (call->callee()) {
      // Call one of the functions generated to implement reduction.
      const auto inputs = getCallInputs(call, interpreter);
      auto ret = ReductionInterpreter::run(executor_id, call->callee(), inputs);
      if (call->type() != Type::Void) {
        // Assign the returned value.
        interpreter->setVar(call, ret);
      }
    } else {
      // Call an internal runtime function.
      const auto func_ptr = bindStub(executor_id, call);
      const auto inputs = getCallInputs(call, interpreter);
      ReductionInterpreter::EvalValue ret;
      func_ptr(&ret, &inputs);
      if (call->type() != Type::Void) {
        // Assign the returned value.
        interpreter->setVar(call, ret);
      }
    }
    return;
  }

  static void runExternalCall(const Instruction* instruction,
                              ReductionInterpreterImpl* interpreter) {
    auto executor_id = interpreter->getExecutorId();
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto external_call = static_cast<const ExternalCall*>(instruction);
    const auto& arguments = external_call->arguments();
    const auto argument_types = get_value_types(arguments);
    const auto func_ptr = bindStub(executor_id, external_call);
    const auto inputs = getCallInputs(external_call, interpreter);
    ReductionInterpreter::EvalValue output;
    func_ptr(&output, &inputs);
    interpreter->setVar(external_call, output);
  }

  static void runAlloca(const Instruction* instruction,
                        ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto alloca = static_cast<const Alloca*>(instruction);
    const auto element_size = get_element_size(alloca->type());
    CHECK(is_integer_type(alloca->array_size()->type()));
    const auto array_size = interpreter->vars_[alloca->array_size()->id()];
    interpreter->alloca_buffers_.emplace_back(element_size * array_size.int_val);
    ReductionInterpreter::EvalValue eval_value;
    eval_value.mutable_ptr = interpreter->alloca_buffers_.back().data();
    interpreter->setVar(alloca, eval_value);
  }

  static void runMemCpy(const Instruction* instruction,
                        ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto memcpy = static_cast<const MemCpy*>(instruction);
    CHECK(is_pointer_type(memcpy->dest()->type()));
    CHECK(is_pointer_type(memcpy->source()->type()));
    CHECK(is_integer_type(memcpy->size()->type()));
    const auto dest = interpreter->vars_[memcpy->dest()->id()];
    const auto source = interpreter->vars_[memcpy->source()->id()];
    const auto size = interpreter->vars_[memcpy->size()->id()];
    ::memcpy(dest.mutable_ptr, source.ptr, size.int_val);
  }

  static void runReturnEarly(const Instruction* instruction,
                             ReductionInterpreterImpl* interpreter) {
    CHECK(!interpreter->ret_) << "Function has already returned";
    const auto ret_early = static_cast<const ReturnEarly*>(instruction);
    CHECK(ret_early->cond()->type() == Type::Int1);
    const auto cond = interpreter->vars_[ret_early->cond()->id()];

    auto error_code = ret_early->error_code();

    if (cond.int_val) {
      auto rc = interpreter->vars_[error_code->id()].int_val;
      interpreter->ret_ = ReductionInterpreter::MakeEvalValue(rc);
    }
  }

  static void runFor(const Instruction* instruction,
                     ReductionInterpreterImpl* interpreter) {
    auto executor_id = interpreter->getExecutorId();
    CHECK(!interpreter->ret_) << "Function has already returned";
    const size_t saved_alloca_count = interpreter->alloca_buffers_.size();
    const auto for_loop = static_cast<const For*>(instruction);
    CHECK(is_integer_type(for_loop->start()->type()));
    CHECK(is_integer_type(for_loop->end()->type()));
    const auto start = interpreter->vars_[for_loop->start()->id()];
    const auto end = interpreter->vars_[for_loop->end()->id()];
    for (int64_t i = start.int_val; i < end.int_val; ++i) {
      // The start and end indices are absolute, but the iteration happens from 0.
      // Subtract the start index before setting the iterator.
      interpreter->vars_[for_loop->iter()->id()].int_val = i - start.int_val;
      auto ret =
          ReductionInterpreter::run(executor_id, for_loop->body(), interpreter->vars_);
      if (ret) {
        interpreter->ret_ = *ret;
        break;
      }
    }
    // Pop all the alloca buffers allocated by the code in the loop.
    interpreter->alloca_buffers_.resize(saved_alloca_count);
  }

 private:
  // Set the variable based on its id.
  void setVar(const Value* var, ReductionInterpreter::EvalValue value) {
    vars_[var->id()] = value;
  }

  // Seed the parameters of the callee.
  template <class Call>
  static std::vector<ReductionInterpreter::EvalValue> getCallInputs(
      const Call* call,
      const ReductionInterpreterImpl* interpreter) {
    std::vector<ReductionInterpreter::EvalValue> inputs;
    inputs.reserve(interpreter->vars_.size());
    for (const auto argument : call->arguments()) {
      inputs.push_back(interpreter->vars_[argument->id()]);
    }
    return inputs;
  }

  // Bind and cache a stub call.
  template <class Call>
  static StubGenerator::Stub bindStub(const size_t executor_id, const Call* call) {
    const auto func_ptr =
        call->cached_callee()
            ? reinterpret_cast<StubGenerator::Stub>(call->cached_callee())
            : StubGenerator::generateStub(executor_id,
                                          call->callee_name(),
                                          get_value_types(call->arguments()),
                                          call->type(),
                                          call->external());
    CHECK(func_ptr);
    call->set_cached_callee(reinterpret_cast<void*>(func_ptr));
    return func_ptr;
  }

  // Holds executor id
  size_t executor_id_;
  // Holds the evaluated values.
  std::vector<ReductionInterpreter::EvalValue> vars_;
  // Holds buffers allocated by the alloca instruction.
  std::vector<std::vector<int8_t>> alloca_buffers_;
  // Holds the value returned by the function.
  std::optional<ReductionInterpreter::EvalValue> ret_ = std::nullopt;
};

void GetElementPtr::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runGetElementPtr(this, interpreter);
}

void Load::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runLoad(this, interpreter);
}

void ICmp::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runICmp(this, interpreter);
}

void BinaryOperator::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runBinaryOperator(this, interpreter);
}

void Cast::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runCast(this, interpreter);
}

void Ret::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runRet(this, interpreter);
}

void Call::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runCall(this, interpreter);
}

void ExternalCall::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runExternalCall(this, interpreter);
}

void Alloca::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runAlloca(this, interpreter);
}

void MemCpy::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runMemCpy(this, interpreter);
}

void ReturnEarly::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runReturnEarly(this, interpreter);
}

void For::run(ReductionInterpreterImpl* interpreter) {
  ReductionInterpreterImpl::runFor(this, interpreter);
}

namespace {

// Create an evaluated constant.
ReductionInterpreter::EvalValue eval_constant(const Constant* constant) {
  switch (constant->type()) {
    case Type::Int8:
    case Type::Int32:
    case Type::Int64: {
      return ReductionInterpreter::MakeEvalValue(
          static_cast<const ConstantInt*>(constant)->value());
    }
    case Type::Float: {
      return ReductionInterpreter::MakeEvalValue(
          static_cast<float>(static_cast<const ConstantFP*>(constant)->value()));
    }
    case Type::Double: {
      return ReductionInterpreter::MakeEvalValue(
          static_cast<const ConstantFP*>(constant)->value());
    }
    default: {
      LOG(FATAL) << "Constant type not supported: " << static_cast<int>(constant->type());
      break;
    }
  }
  return {};
}

}  // namespace

ReductionInterpreter::EvalValue ReductionInterpreter::run(
    const size_t executor_id,
    const Function* function,
    const std::vector<ReductionInterpreter::EvalValue>& inputs) {
  const auto last_id = function->body().back()->id();
  const auto& arg_types = function->arg_types();
  std::vector<ReductionInterpreter::EvalValue> vars(last_id + 1);
  // Add the arguments to the variable map.
  for (size_t i = 0; i < arg_types.size(); ++i) {
    vars[function->arg(i)->id()] = inputs[i];
  }
  // Add constants to the variable map.
  for (const auto& constant : function->constants()) {
    vars[constant->id()] = eval_constant(constant.get());
  }
  const auto maybe_ret = run(executor_id, function->body(), vars);
  CHECK(maybe_ret);
  return *maybe_ret;
}

std::optional<ReductionInterpreter::EvalValue> ReductionInterpreter::run(
    const size_t executor_id,
    const std::vector<std::unique_ptr<Instruction>>& body,
    const std::vector<ReductionInterpreter::EvalValue>& vars) {
  ReductionInterpreterImpl interp_impl(executor_id, vars);
  for (const auto& instr : body) {
    instr->run(&interp_impl);
    const auto ret = interp_impl.ret();
    if (ret) {
      return *ret;
    }
  }
  return interp_impl.ret();
}
