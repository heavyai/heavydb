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

#include <string>
#include <vector>

#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "Shared/toString.h"
#include "TableFunctionOutputBufferSizeType.h"

namespace table_functions {

struct TableFunctionOutputRowSizer {
  OutputBufferSizeType type{OutputBufferSizeType::kConstant};
  const size_t val{0};

 public:
  std::string toString() const {
    switch (type) {
      case OutputBufferSizeType::kUserSpecifiedConstantParameter:
        return "kUserSpecifiedConstantParameter[" + std::to_string(val) + "]";
      case OutputBufferSizeType::kUserSpecifiedRowMultiplier:
        return "kUserSpecifiedRowMultiplier[" + std::to_string(val) + "]";
      case OutputBufferSizeType::kConstant:
        return "kConstant[" + std::to_string(val) + "]";
    }
    return "";
  }
};

class TableFunction {
 public:
  TableFunction(const std::string& name,
                const TableFunctionOutputRowSizer output_sizer,
                const std::vector<ExtArgumentType>& input_args,
                const std::vector<ExtArgumentType>& output_args,
                bool is_runtime)
      : name_(name)
      , output_sizer_(output_sizer)
      , input_args_(input_args)
      , output_args_(output_args)
      , is_runtime_(is_runtime) {}

  std::vector<ExtArgumentType> getArgs() const {
    std::vector<ExtArgumentType> args;
    args.insert(args.end(), input_args_.begin(), input_args_.end());
    args.insert(args.end(), output_args_.begin(), output_args_.end());
    return args;
  }
  const std::vector<ExtArgumentType>& getInputArgs() const { return input_args_; }
  const ExtArgumentType getRet() const { return ExtArgumentType::Int32; }

  SQLTypeInfo getInputSQLType(const size_t idx) const;
  SQLTypeInfo getOutputSQLType(const size_t idx) const;

  auto getInputsSize() const { return input_args_.size(); }
  auto getOutputsSize() const { return output_args_.size(); }

  auto getName() const { return name_; }

  bool hasNonUserSpecifiedOutputSizeConstant() const {
    return output_sizer_.type == OutputBufferSizeType::kConstant;
  }

  bool hasUserSpecifiedOutputSizeConstant() const {
    return output_sizer_.type == OutputBufferSizeType::kUserSpecifiedConstantParameter;
  }

  bool hasUserSpecifiedOutputSizeMultiplier() const {
    return output_sizer_.type == OutputBufferSizeType::kUserSpecifiedRowMultiplier;
  }

  OutputBufferSizeType getOutputRowSizeType() const { return output_sizer_.type; }

  size_t getOutputRowSizeParameter() const { return output_sizer_.val; }

  bool isRuntime() const { return is_runtime_; }

  bool isGPU() const { return (name_.find("_cpu_") == std::string::npos); }

  bool isCPU() const { return (name_.find("_gpu_") == std::string::npos); }

  std::string toString() const {
    auto result = "TableFunction(" + name_ + ", [";
    result += ExtensionFunctionsWhitelist::toString(input_args_);
    result += "], [";
    result += ExtensionFunctionsWhitelist::toString(output_args_);
    result += "], is_runtime=" + std::string((is_runtime_ ? "true" : "false"));
    result += ", sizer=" + ::toString(output_sizer_);
    result += ")";
    return result;
  }

  std::string toStringSQL() const {
    auto result = name_ + "(";
    result += ExtensionFunctionsWhitelist::toStringSQL(input_args_);
    result += ") -> (";
    result += ExtensionFunctionsWhitelist::toStringSQL(output_args_);
    result += ")";
    return result;
  }

 private:
  const std::string name_;
  const TableFunctionOutputRowSizer output_sizer_;
  const std::vector<ExtArgumentType> input_args_;
  const std::vector<ExtArgumentType> output_args_;
  const bool is_runtime_;
};

class TableFunctionsFactory {
 public:
  static void add(const std::string& name,
                  const TableFunctionOutputRowSizer sizer,
                  const std::vector<ExtArgumentType>& input_args,
                  const std::vector<ExtArgumentType>& output_args,
                  bool is_runtime = false);

  static std::vector<TableFunction> get_table_funcs(const std::string& name,
                                                    const bool is_gpu);
  static void init();
  static void reset();

 private:
  static std::unordered_map<std::string, TableFunction> functions_;

  friend class ::ExtensionFunctionsWhitelist;
};

}  // namespace table_functions
