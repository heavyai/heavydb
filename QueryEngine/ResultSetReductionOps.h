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

#pragma once

#include <Shared/Logger.h>

#include <memory>
#include <string>
#include <vector>

extern thread_local size_t g_value_id;

// A collection of operators heavily inspired from LLVM IR which are both easy to
// translated to LLVM IR and interpreted, for small result sets, to avoid compilation
// overhead. In order to keep things simple, there is no general-purpose control flow.
// Instead, there is ReturnEarly for early return from a function based on a logical
// condition and For, which iterates between a start and an index and executes the body.

enum class Type {
  Int1,
  Int8,
  Int32,
  Int64,
  Float,
  Double,
  Void,
  Int8Ptr,
  Int32Ptr,
  Int64Ptr,
  FloatPtr,
  DoublePtr,
  VoidPtr,
  Int64PtrPtr,
};

// Retrieves the type a pointer type points to.
inline Type pointee_type(const Type pointer) {
  switch (pointer) {
    case Type::Int8Ptr: {
      return Type::Int8;
    }
    case Type::Int32Ptr: {
      return Type::Int32;
    }
    case Type::Int64Ptr: {
      return Type::Int64;
    }
    case Type::FloatPtr: {
      return Type::Float;
    }
    case Type::DoublePtr: {
      return Type::Double;
    }
    case Type::Int64PtrPtr: {
      return Type::Int64Ptr;
    }
    default: {
      LOG(FATAL) << "Invalid pointer type: " << static_cast<int>(pointer);
    }
  }
  return Type::Void;
}

// Creates a pointer type from the given type.
inline Type pointer_type(const Type pointee) {
  switch (pointee) {
    case Type::Int8: {
      return Type::Int8Ptr;
    }
    case Type::Int64: {
      return Type::Int64Ptr;
    }
    case Type::Int64Ptr: {
      return Type::Int64PtrPtr;
    }
    default: {
      LOG(FATAL) << "Invalid pointee type: " << static_cast<int>(pointee);
    }
  }
  return Type::Void;
}

class Value {
 public:
  Value(const Type type, const std::string& label)
      : type_(type), label_(label), id_(g_value_id++) {}

  Type type() const { return type_; }

  size_t id() const { return id_; }

  const std::string& label() const { return label_; }

  virtual ~Value() = default;

 private:
  const Type type_;
  // The label of the value, useful for debugging the generated LLVM IR.
  const std::string label_;
  // An unique id, starting from 0, relative to the function. Used by the interpreter to
  // implement a dense map of evaluated values.
  const size_t id_;
};

class Constant : public Value {
 public:
  Constant(const Type type) : Value(type, "") {}
};

class ConstantInt : public Constant {
 public:
  ConstantInt(const int64_t value, const Type target) : Constant(target), value_(value) {}

  int64_t value() const { return value_; }

 private:
  const int64_t value_;
};

class ConstantFP : public Constant {
 public:
  ConstantFP(const double value, const Type target) : Constant(target), value_(value) {}

  double value() const { return value_; }

 private:
  const double value_;
};

class Argument : public Value {
 public:
  Argument(const Type type, const std::string& label) : Value(type, label) {}
};

class ReductionInterpreterImpl;

class Instruction : public Value {
 public:
  Instruction(const Type type, const std::string& label) : Value(type, label) {}

  // Run the instruction in the given interpreter.
  virtual void run(ReductionInterpreterImpl* interpreter) = 0;
};

// A function, defined by its signature and instructions, which it owns.
class Function {
 public:
  struct NamedArg {
    std::string name;
    Type type;
  };

  Function(const std::string name,
           const std::vector<NamedArg>& arg_types,
           const Type ret_type,
           const bool always_inline)
      : name_(name)
      , arg_types_(arg_types)
      , ret_type_(ret_type)
      , always_inline_(always_inline) {
    g_value_id = 0;
    for (const auto& named_arg : arg_types_) {
      arguments_.emplace_back(new Argument(named_arg.type, named_arg.name));
    }
  }

  const std::string& name() const { return name_; }

  const std::vector<NamedArg>& arg_types() const { return arg_types_; }

  Argument* arg(const size_t idx) const { return arguments_[idx].get(); }

  Type ret_type() const { return ret_type_; }

  const std::vector<std::unique_ptr<Instruction>>& body() const { return body_; }

  const std::vector<std::unique_ptr<Constant>>& constants() const { return constants_; }

  bool always_inline() const { return always_inline_; }

  template <typename Tp, typename... Args>
  Value* add(Args&&... args) {
    body_.emplace_back(new Tp(std::forward<Args>(args)...));
    return body_.back().get();
  }

  template <typename Tp, typename... Args>
  Value* addConstant(Args&&... args) {
    constants_.emplace_back(new Tp(std::forward<Args>(args)...));
    return constants_.back().get();
  }

 private:
  const std::string name_;
  const std::vector<NamedArg> arg_types_;
  const Type ret_type_;
  std::vector<std::unique_ptr<Instruction>> body_;
  const bool always_inline_;
  std::vector<std::unique_ptr<Argument>> arguments_;
  std::vector<std::unique_ptr<Constant>> constants_;
};

class GetElementPtr : public Instruction {
 public:
  GetElementPtr(const Value* base, const Value* index, const std::string& label)
      : Instruction(base->type(), label), base_(base), index_(index) {}

  const Value* base() const { return base_; }

  const Value* index() const { return index_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const Value* base_;
  const Value* index_;
};

class Load : public Instruction {
 public:
  Load(const Value* source, const std::string& label)
      : Instruction(pointee_type(source->type()), label), source_(source) {}

  const Value* source() const { return source_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const Value* source_;
};

class ICmp : public Instruction {
 public:
  enum class Predicate {
    NE,
    EQ,
  };

  ICmp(const Predicate predicate,
       const Value* lhs,
       const Value* rhs,
       const std::string& label)
      : Instruction(Type::Int1, label), predicate_(predicate), lhs_(lhs), rhs_(rhs) {}

  Predicate predicate() const { return predicate_; }

  const Value* lhs() const { return lhs_; }

  const Value* rhs() const { return rhs_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const Predicate predicate_;
  const Value* lhs_;
  const Value* rhs_;
};

class BinaryOperator : public Instruction {
 public:
  enum class BinaryOp {
    Add,
    Mul,
  };

  BinaryOperator(const BinaryOp op,
                 const Value* lhs,
                 const Value* rhs,
                 const std::string& label)
      : Instruction(Type::Int1, label), op_(op), lhs_(lhs), rhs_(rhs) {}

  BinaryOp op() const { return op_; }

  const Value* lhs() const { return lhs_; }

  const Value* rhs() const { return rhs_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const BinaryOp op_;
  const Value* lhs_;
  const Value* rhs_;
};

class Cast : public Instruction {
 public:
  enum class CastOp {
    Trunc,
    SExt,
    BitCast,
  };

  Cast(const CastOp op, const Value* source, const Type type, const std::string& label)
      : Instruction(type, label), op_(op), source_(source) {}

  CastOp op() const { return op_; }

  const Value* source() const { return source_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const CastOp op_;
  const Value* source_;
};

class Ret : public Instruction {
 public:
  Ret(const Value* value) : Instruction(value->type(), ""), value_(value) {}

  Ret() : Instruction(Type::Void, ""), value_(nullptr) {}

  const Value* value() const { return value_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const Value* value_;
};

// An internal runtime function. In this context, internal means either part of the
// bitcode runtime (given by name) or one of the reduction functions.
class Call : public Instruction {
 public:
  Call(const Function* callee,
       const std::vector<const Value*>& arguments,
       const std::string& label)
      : Instruction(callee->ret_type(), label)
      , callee_(callee)
      , arguments_(arguments)
      , cached_callee_(nullptr) {}

  Call(const std::string& callee_name,
       const std::vector<const Value*>& arguments,
       const std::string& label)
      : Instruction(Type::Void, label)
      , callee_name_(callee_name)
      , callee_(nullptr)
      , arguments_(arguments)
      , cached_callee_(nullptr) {}

  bool external() const { return false; }

  const std::string& callee_name() const { return callee_name_; }

  const Function* callee() const { return callee_; }

  const std::vector<const Value*>& arguments() const { return arguments_; }

  void run(ReductionInterpreterImpl* interpreter) override;

  void* cached_callee() const { return cached_callee_; }

  void set_cached_callee(void* cached_callee) const { cached_callee_ = cached_callee; }

 private:
  const std::string callee_name_;
  const Function* callee_;
  const std::vector<const Value*> arguments_;
  // For performance reasons, the pointer of the native function is stored in this field.
  mutable void* cached_callee_;
};

// An external runtime function, with C binding.
class ExternalCall : public Instruction {
 public:
  ExternalCall(const std::string& callee_name,
               const Type ret_type,
               const std::vector<const Value*>& arguments,
               const std::string& label)
      : Instruction(ret_type, label)
      , callee_name_(callee_name)
      , ret_type_(ret_type)
      , arguments_(arguments)
      , cached_callee_(nullptr) {}

  bool external() const { return true; }

  const std::string& callee_name() const { return callee_name_; }

  const std::vector<const Value*>& arguments() const { return arguments_; }

  void run(ReductionInterpreterImpl* interpreter) override;

  void* cached_callee() const { return cached_callee_; }

  void set_cached_callee(void* cached_callee) const { cached_callee_ = cached_callee; }

 private:
  const std::string callee_name_;
  const Type ret_type_;
  const std::vector<const Value*> arguments_;
  mutable void* cached_callee_;
};

class Alloca : public Instruction {
 public:
  Alloca(const Type element_type, const Value* array_size, const std::string& label)
      : Instruction(pointer_type(element_type), label), array_size_(array_size) {}

  const Value* array_size() const { return array_size_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const Value* array_size_;
};

class MemCpy : public Instruction {
 public:
  MemCpy(const Value* dest, const Value* source, const Value* size)
      : Instruction(Type::Void, ""), dest_(dest), source_(source), size_(size) {}

  const Value* dest() const { return dest_; }

  const Value* source() const { return source_; }

  const Value* size() const { return size_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const Value* dest_;
  const Value* source_;
  const Value* size_;
};

// Returns from the current function with the given error code, if the provided condition
// is true. If the function return type is void, the error code is ignored.
class ReturnEarly : public Instruction {
 public:
  ReturnEarly(const Value* cond, const int error_code, const std::string& label)
      : Instruction(Type::Void, label), cond_(cond), error_code_(error_code) {}

  const Value* cond() const { return cond_; }

  int error_code() const { return error_code_; }

  void run(ReductionInterpreterImpl* interpreter) override;

 private:
  const Value* cond_;
  const int error_code_;
};

// An operation which executes the provided body from the given start index to the end
// index (exclusive). Additionally, the iterator is added to the variables seen by the
// body.
class For : public Instruction {
 public:
  For(const Value* start, const Value* end, const std::string& label)
      : Instruction(Type::Void, label)
      , start_(start)
      , end_(end)
      , iter_(Type::Int64, label) {}

  const std::vector<std::unique_ptr<Instruction>>& body() const { return body_; }

  const Value* start() const { return start_; }

  const Value* end() const { return end_; }

  const Value* iter() const { return &iter_; }

  void run(ReductionInterpreterImpl* interpreter) override;

  template <typename Tp, typename... Args>
  Value* add(Args&&... args) {
    body_.emplace_back(new Tp(std::forward<Args>(args)...));
    return body_.back().get();
  }

 private:
  std::vector<std::unique_ptr<Instruction>> body_;
  const Value* start_;
  const Value* end_;
  // Since the iterator always moves between the start and the end, just store a dummy
  // value. During codegen or interpretation, it will be mapped to the current value of
  // the iterator.
  const Value iter_;
};
