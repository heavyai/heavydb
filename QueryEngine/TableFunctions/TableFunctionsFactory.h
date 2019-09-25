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

namespace table_functions {

enum class OutputBufferSizeType {
  kUserSpecifiedConstantParameter,
  kUserSpecifiedRowMultiplier,
  kConstant
};

struct TableFunctionOutputRowSizer {
  OutputBufferSizeType type{OutputBufferSizeType::kConstant};
  const size_t val{0};
};

class TableFunction {
 public:
  TableFunction(const std::string& name,
                const TableFunctionOutputRowSizer output_sizer,
                const std::vector<ExtArgumentType>& input_args,
                const std::vector<ExtArgumentType>& output_args)
      : name_(name)
      , output_sizer_(output_sizer)
      , input_args_(input_args)
      , output_args_(output_args) {}

  std::vector<ExtArgumentType> getArgs() const {
    std::vector<ExtArgumentType> args;
    args.insert(args.end(), input_args_.begin(), input_args_.end());
    args.insert(args.end(), output_args_.begin(), output_args_.end());
    return args;
  }

  SQLTypeInfo getOutputSQLType(const size_t idx) const;

  auto getOutputsSize() const { return output_args_.size(); }

  auto getName() const { return name_; }

  bool hasUserSpecifiedOutputMultiplier() const {
    return output_sizer_.type == OutputBufferSizeType::kUserSpecifiedRowMultiplier;
  }

  size_t getOutputRowParameter() const { return output_sizer_.val; }

 private:
  const std::string name_;
  const TableFunctionOutputRowSizer output_sizer_;
  const std::vector<ExtArgumentType> input_args_;
  const std::vector<ExtArgumentType> output_args_;
};

class TableFunctionsFactory {
 public:
  static void add(const std::string& name,
                  const TableFunctionOutputRowSizer sizer,
                  const std::vector<ExtArgumentType>& input_args,
                  const std::vector<ExtArgumentType>& output_args);

  static const TableFunction& get(const std::string& name);

 private:
  static void init();

  static std::unordered_map<std::string, TableFunction> functions_;

  friend class ::ExtensionFunctionsWhitelist;
};

}  // namespace table_functions
