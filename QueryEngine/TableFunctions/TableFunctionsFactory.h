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
#include "QueryEngine/TableFunctionHelper.h"
#include "Shared/toString.h"
#include "TableFunctionOutputBufferSizeType.h"

#define DEFAULT_ROW_MULTIPLIER_SUFFIX "__default_RowMultiplier_"
#define DEFAULT_ROW_MULTIPLIER_VALUE 1

/*

  TableFunction represents a User-Defined Table Function (UDTF) and it
  holds the following information:

  - the name of a table function that corresponds to its
    implementation. The name must match the following pattern:

    \w[\w\d_]*([_][_](gpu_|cpu_|)\d*|)

    where the first part left to the double underscore is the
    so-called SQL name of table function that is used in SQL query
    context, and the right part determines a particular implementation
    of the table function. One can define many implementations for the
    same SQL table function with specializations to

    + different argument types (overloading support)

    + different execution context, CPU or GPU. When gpu or cpu is not
      present, the implementation is assumed to be valid for both CPU
      and GPU contexts.

  - the output sizer parameter <sizer> that determines the allocated
    size of the output columns:

    + UserSpecifiedRowMultiplier - the allocated column size will be

        <sizer value> * <size of the input columns>

      where <sizer value> is user-specified integer value as specified
      in the <sizer> argument position of the table function call.

    + UserSpecifiedConstantParameter - the allocated column size will
      be user-specified integer value as specified in the <sizer>
      argument position of the table function call.

    + Constant - the allocated output column size will be <sizer>.

    + TableFunctionSpecifiedParameter - The table function
      implementation must call resize to allocate output column
      buffers. The <sizer> value is not used.

    The actual size of the output column is returned by the table
    function implementation that must be equal or smaller to the
    allocated output column size.

  - the list of input argument types. The input argument type can be a
    scalar or a column type (that is `Column<scalar>`). Supported
    scalar types are int8, ..., int64, double, float, bool.

  - the list of output argument types. The output types of table
    functions is always some column type. Hence, the output argument
    types are stored as scalar types that correspond to the data type
    of the output columns.

  - a boolean flag specifying the table function is a load-time or
    run-time function. Run-time functions can be overwitten or removed
    by users. Load-time functions cannot be redefined in run-time.

  Future notes:

  - introduce a list of output column names. Currently, the names of
    output columns match the pattern

      out\d+

    but for better UX it would be nice to enable user-defined names
    for output columns.

 */

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
      case OutputBufferSizeType::kTableFunctionSpecifiedParameter:
        return "kTableFunctionSpecifiedParameter[" + std::to_string(val) + "]";
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
                const std::vector<ExtArgumentType>& sql_args,
                const std::vector<std::map<std::string, std::string>>& annotations,
                bool is_runtime,
                bool uses_manager)
      : name_(name)
      , output_sizer_(output_sizer)
      , input_args_(input_args)
      , output_args_(output_args)
      , sql_args_(sql_args)
      , annotations_(annotations)
      , is_runtime_(is_runtime)
      , uses_manager_(uses_manager) {}

  std::vector<ExtArgumentType> getArgs(const bool ensure_column = false) const {
    std::vector<ExtArgumentType> args;
    args.insert(args.end(), input_args_.begin(), input_args_.end());
    if (ensure_column) {
      // map row dtype to column type
      std::for_each(output_args_.begin(), output_args_.end(), [&args](auto t) {
        args.push_back(ext_arg_type_ensure_column(t));
      });
    } else {
      args.insert(args.end(), output_args_.begin(), output_args_.end());
    }
    return args;
  }
  const std::vector<ExtArgumentType>& getInputArgs() const { return input_args_; }
  const std::vector<ExtArgumentType>& getOutputArgs() const { return output_args_; }
  const std::vector<ExtArgumentType>& getSqlArgs() const { return sql_args_; }
  const std::vector<std::map<std::string, std::string>>& getAnnotations() const {
    return annotations_;
  }
  const ExtArgumentType getRet() const { return ExtArgumentType::Int32; }

  SQLTypeInfo getInputSQLType(const size_t idx) const;
  SQLTypeInfo getOutputSQLType(const size_t idx) const;

  int32_t countScalarArgs() const;

  auto getInputsSize() const { return input_args_.size(); }
  auto getOutputsSize() const { return output_args_.size(); }

  std::string getName(const bool drop_suffix = false, const bool lower = false) const;

  auto getSignature() const {
    return getName(/*drop_suffix=*/true, /*lower=*/true) + "(" +
           ExtensionFunctionsWhitelist::toString(input_args_) + ")";
  }

  bool hasCompileTimeOutputSizeConstant() const {
    return output_sizer_.type == OutputBufferSizeType::kConstant;
  }

  bool hasUserSpecifiedOutputSizeConstant() const {
    return output_sizer_.type == OutputBufferSizeType::kUserSpecifiedConstantParameter;
  }

  bool hasConstantOutputSize() const {
    return output_sizer_.type == OutputBufferSizeType::kConstant ||
           output_sizer_.type == OutputBufferSizeType::kUserSpecifiedConstantParameter;
  }

  bool hasTableFunctionSpecifiedParameter() const {
    return output_sizer_.type == OutputBufferSizeType::kTableFunctionSpecifiedParameter;
  }

  bool hasUserSpecifiedOutputSizeParameter() const {
    return output_sizer_.type == OutputBufferSizeType::kUserSpecifiedConstantParameter ||
           output_sizer_.type == OutputBufferSizeType::kUserSpecifiedRowMultiplier;
  }

  bool hasNonUserSpecifiedOutputSize() const {
    return output_sizer_.type == OutputBufferSizeType::kConstant ||
           output_sizer_.type == OutputBufferSizeType::kTableFunctionSpecifiedParameter;
  }

  bool hasOutputSizeIndependentOfInputSize() const {
    return output_sizer_.type == OutputBufferSizeType::kConstant ||
           output_sizer_.type == OutputBufferSizeType::kUserSpecifiedConstantParameter ||
           output_sizer_.type == OutputBufferSizeType::kTableFunctionSpecifiedParameter;
  }

  bool hasOutputSizeKnownPreLaunch() const {
    return output_sizer_.type == OutputBufferSizeType::kConstant ||
           output_sizer_.type == OutputBufferSizeType::kUserSpecifiedConstantParameter ||
           output_sizer_.type == OutputBufferSizeType::kUserSpecifiedRowMultiplier;
  }

  bool hasUserSpecifiedOutputSizeMultiplier() const {
    return output_sizer_.type == OutputBufferSizeType::kUserSpecifiedRowMultiplier;
  }

  OutputBufferSizeType getOutputRowSizeType() const { return output_sizer_.type; }

  size_t getOutputRowSizeParameter() const { return output_sizer_.val; }

  const std::map<std::string, std::string>& getAnnotation(const size_t idx) const;
  const std::map<std::string, std::string>& getInputAnnotation(
      const size_t input_arg_idx) const;
  const std::map<std::string, std::string>& getOutputAnnotation(
      const size_t output_arg_idx) const;

  std::pair<int32_t, int32_t> getInputID(const size_t idx) const;

  size_t getSqlOutputRowSizeParameter() const;

  size_t getOutputRowSizeParameter(const std::vector<SQLTypeInfo>& variant) const {
    auto val = output_sizer_.val;
    if (hasUserSpecifiedOutputSizeMultiplier()) {
      size_t col_index = 0;
      size_t func_arg_index = 0;
      for (const auto& ti : variant) {
        func_arg_index++;
        if (ti.is_column_list()) {
          col_index += ti.get_dimension();
        } else {
          col_index++;
        }
        if (func_arg_index == val) {
          val = col_index;
          break;
        }
      }
    }
    return val;
  }

  bool isRuntime() const { return is_runtime_; }

  bool usesManager() const { return uses_manager_; }

  inline bool isGPU() const {
    return !usesManager() && (name_.find("_cpu_", name_.find("__")) == std::string::npos);
  }

  inline bool isCPU() const {
    return usesManager() || (name_.find("_gpu_", name_.find("__")) == std::string::npos);
  }

  inline bool useDefaultSizer() const {
    // Functions that use a default sizer value have one less argument
    return (name_.find("_default_", name_.find("__")) != std::string::npos);
  }

  std::string toString() const {
    auto result = "TableFunction(" + name_ + ", input_args=[";
    result += ExtensionFunctionsWhitelist::toString(input_args_);
    result += "], output_args=[";
    result += ExtensionFunctionsWhitelist::toString(output_args_);
    result += "], sql_args=[";
    result += ExtensionFunctionsWhitelist::toString(sql_args_);
    result += "], is_runtime=" + std::string((is_runtime_ ? "true" : "false"));
    result += ", uses_manager=" + std::string((uses_manager_ ? "true" : "false"));
    result += ", sizer=" + ::toString(output_sizer_);
    result += ", annotations=[";
    for (auto annotation : annotations_) {
      if (annotation.empty()) {
        result += "{}, ";
      } else {
        result += "{";
        for (auto it : annotation) {
          result += ::toString(it.first) + ": " + ::toString(it.second);
        }
        result += "}, ";
      }
    }
    result += "])";
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
  const std::vector<ExtArgumentType> sql_args_;
  const std::vector<std::map<std::string, std::string>> annotations_;
  const bool is_runtime_;
  const bool uses_manager_;
};

class TableFunctionsFactory {
 public:
  static void add(const std::string& name,
                  const TableFunctionOutputRowSizer sizer,
                  const std::vector<ExtArgumentType>& input_args,
                  const std::vector<ExtArgumentType>& output_args,
                  const std::vector<ExtArgumentType>& sql_args,
                  const std::vector<std::map<std::string, std::string>>& annotations,
                  bool is_runtime = false,
                  bool uses_manager = false);

  static std::vector<TableFunction> get_table_funcs(const std::string& name,
                                                    const bool is_gpu);
  static std::vector<TableFunction> get_table_funcs(const bool is_runtime = false);
  static void init();
  static void reset();

 private:
  static std::unordered_map<std::string, TableFunction> functions_;

  friend class ::ExtensionFunctionsWhitelist;
};

}  // namespace table_functions
