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

#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"

#include <boost/algorithm/string.hpp>
#include <mutex>
#include <unordered_set>

extern bool g_enable_table_functions;
extern bool g_enable_dev_table_functions;

namespace table_functions {

namespace {

SQLTypeInfo ext_arg_pointer_type_to_type_info(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::PInt8:
      return SQLTypeInfo(kTINYINT, false);
    case ExtArgumentType::PInt16:
      return SQLTypeInfo(kSMALLINT, false);
    case ExtArgumentType::PInt32:
      return SQLTypeInfo(kINT, false);
    case ExtArgumentType::PInt64:
      return SQLTypeInfo(kBIGINT, false);
    case ExtArgumentType::PFloat:
      return SQLTypeInfo(kFLOAT, false);
    case ExtArgumentType::PDouble:
      return SQLTypeInfo(kDOUBLE, false);
    case ExtArgumentType::PBool:
      return SQLTypeInfo(kBOOLEAN, false);
    case ExtArgumentType::ColumnInt8:
      return generate_column_type(kTINYINT);
    case ExtArgumentType::ColumnInt16:
      return generate_column_type(kSMALLINT);
    case ExtArgumentType::ColumnInt32:
      return generate_column_type(kINT);
    case ExtArgumentType::ColumnInt64:
      return generate_column_type(kBIGINT);
    case ExtArgumentType::ColumnFloat:
      return generate_column_type(kFLOAT);
    case ExtArgumentType::ColumnDouble:
      return generate_column_type(kDOUBLE);
    case ExtArgumentType::ColumnBool:
      return generate_column_type(kBOOLEAN);
    case ExtArgumentType::ColumnListInt8:
      return generate_column_list_type(kTINYINT);
    case ExtArgumentType::ColumnListInt16:
      return generate_column_list_type(kSMALLINT);
    case ExtArgumentType::ColumnListInt32:
      return generate_column_list_type(kINT);
    case ExtArgumentType::ColumnListInt64:
      return generate_column_list_type(kBIGINT);
    case ExtArgumentType::ColumnListFloat:
      return generate_column_list_type(kFLOAT);
    case ExtArgumentType::ColumnListDouble:
      return generate_column_list_type(kDOUBLE);
    case ExtArgumentType::ColumnListBool:
      return generate_column_list_type(kBOOLEAN);
    default:
      LOG(WARNING) << "ext_arg_pointer_type_to_type_info: ExtArgumentType `"
                   << ExtensionFunctionsWhitelist::toString(ext_arg_type)
                   << "` conversion to SQLTypeInfo not implemented.";
      UNREACHABLE();
  }
  UNREACHABLE();
  return SQLTypeInfo(kNULLT, false);
}

SQLTypeInfo ext_arg_type_to_type_info_output(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::PInt8:
    case ExtArgumentType::ColumnInt8:
    case ExtArgumentType::ColumnListInt8:
    case ExtArgumentType::Int8:
      return SQLTypeInfo(kTINYINT, false);
    case ExtArgumentType::PInt16:
    case ExtArgumentType::ColumnInt16:
    case ExtArgumentType::ColumnListInt16:
    case ExtArgumentType::Int16:
      return SQLTypeInfo(kSMALLINT, false);
    case ExtArgumentType::PInt32:
    case ExtArgumentType::ColumnInt32:
    case ExtArgumentType::ColumnListInt32:
    case ExtArgumentType::Int32:
      return SQLTypeInfo(kINT, false);
    case ExtArgumentType::PInt64:
    case ExtArgumentType::ColumnInt64:
    case ExtArgumentType::ColumnListInt64:
    case ExtArgumentType::Int64:
      return SQLTypeInfo(kBIGINT, false);
    case ExtArgumentType::PFloat:
    case ExtArgumentType::ColumnFloat:
    case ExtArgumentType::ColumnListFloat:
    case ExtArgumentType::Float:
      return SQLTypeInfo(kFLOAT, false);
    case ExtArgumentType::PDouble:
    case ExtArgumentType::ColumnDouble:
    case ExtArgumentType::ColumnListDouble:
    case ExtArgumentType::Double:
      return SQLTypeInfo(kDOUBLE, false);
    case ExtArgumentType::PBool:
    case ExtArgumentType::ColumnBool:
    case ExtArgumentType::ColumnListBool:
    case ExtArgumentType::Bool:
      return SQLTypeInfo(kBOOLEAN, false);
    case ExtArgumentType::ColumnTextEncodingDict:
    case ExtArgumentType::ColumnListTextEncodingDict:
    case ExtArgumentType::TextEncodingDict:
      return SQLTypeInfo(kTEXT, false, kENCODING_DICT);
    case ExtArgumentType::ColumnTimestamp:
      return SQLTypeInfo(kTIMESTAMP, 9, 0, false);
    default:
      LOG(WARNING) << "ext_arg_type_to_type_info_output: ExtArgumentType `"
                   << ExtensionFunctionsWhitelist::toString(ext_arg_type)
                   << "` conversion to SQLTypeInfo not implemented.";
      UNREACHABLE();
  }
  UNREACHABLE();
  return SQLTypeInfo(kNULLT, false);
}

}  // namespace

SQLTypeInfo TableFunction::getInputSQLType(const size_t idx) const {
  CHECK_LT(idx, input_args_.size());
  return ext_arg_pointer_type_to_type_info(input_args_[idx]);
}

SQLTypeInfo TableFunction::getOutputSQLType(const size_t idx) const {
  CHECK_LT(idx, output_args_.size());
  // TODO(adb): conditionally handle nulls
  return ext_arg_type_to_type_info_output(output_args_[idx]);
}

int32_t TableFunction::countScalarArgs() const {
  int32_t scalar_args = 0;
  for (const auto& ext_arg : input_args_) {
    if (is_ext_arg_type_scalar(ext_arg)) {
      scalar_args += 1;
    }
  }
  return scalar_args;
}

bool TableFunction::containsPreFlightFn() const {
  if (hasPreFlightOutputSizer()) {
    return true;
  }
  // workaround for default args
  for (size_t idx = 0; idx < std::min(input_args_.size(), annotations_.size()); idx++) {
    const auto& ann = getInputAnnotations(idx);
    if (ann.find("require") != ann.end()) {
      return true;
    }
  }
  return false;
}

const std::map<std::string, std::string> TableFunction::getAnnotations(
    const size_t idx) const {
  CHECK_LE(idx, sql_args_.size() + output_args_.size());
  if (annotations_.empty() || idx >= annotations_.size()) {
    static const std::map<std::string, std::string> empty = {};
    return empty;
  }
  return annotations_[idx];
}

const std::map<std::string, std::string> TableFunction::getInputAnnotations(
    const size_t input_arg_idx) const {
  CHECK_LT(input_arg_idx, input_args_.size());
  return getAnnotations(input_arg_idx);
}

const std::string TableFunction::getInputAnnotation(const size_t input_arg_idx,
                                                    const std::string& key,
                                                    const std::string& default_) const {
  const std::map<std::string, std::string> ann = getInputAnnotations(input_arg_idx);
  const auto& it = ann.find(key);
  if (it != ann.end()) {
    return it->second;
  }
  return default_;
}

const std::map<std::string, std::string> TableFunction::getOutputAnnotations(
    const size_t output_arg_idx) const {
  CHECK_LT(output_arg_idx, output_args_.size());
  return getAnnotations(output_arg_idx + sql_args_.size());
}

const std::string TableFunction::getOutputAnnotation(const size_t output_arg_idx,
                                                     const std::string& key,
                                                     const std::string& default_) const {
  const std::map<std::string, std::string> ann = getOutputAnnotations(output_arg_idx);
  const auto& it = ann.find(key);
  if (it != ann.end()) {
    return it->second;
  }
  return default_;
}

const std::map<std::string, std::string> TableFunction::getFunctionAnnotations() const {
  return getAnnotations(sql_args_.size() + output_args_.size());
}

const std::string TableFunction::getFunctionAnnotation(
    const std::string& key,
    const std::string& default_) const {
  const std::map<std::string, std::string> ann = getFunctionAnnotations();
  const auto& it = ann.find(key);
  if (it != ann.end()) {
    return it->second;
  }
  return default_;
}

const std::vector<std::string> TableFunction::getCursorFields(
    const size_t sql_idx) const {
  std::vector<std::string> fields;
  const std::string& line = getInputAnnotation(sql_idx, "fields", "");
  if (line.empty()) {
    static const std::vector<std::string> empty = {};
    return empty;
  }
  std::string substr = line.substr(1, line.size() - 2);
  boost::split(fields, substr, boost::is_any_of(", "), boost::token_compress_on);
  return fields;
}

const std::string TableFunction::getArgTypes(bool use_input_args) const {
  if (use_input_args) {
    std::vector<std::string> arg_types;
    size_t arg_idx = 0;
    for (size_t sql_idx = 0; sql_idx < sql_args_.size(); sql_idx++) {
      const std::vector<std::string> cursor_fields = getCursorFields(sql_idx);
      if (cursor_fields.empty()) {
        // fields => {}
        arg_types.emplace_back(
            ExtensionFunctionsWhitelist::toString(input_args_[arg_idx++]));
      } else {
        std::vector<std::string> vec;
        for (size_t i = 0; i < cursor_fields.size(); i++) {
          vec.emplace_back(ExtensionFunctionsWhitelist::toString(input_args_[arg_idx++]));
        }
        arg_types.emplace_back("Cursor<" + boost::algorithm::join(vec, ", ") + ">");
      }
    }
    return "[" + boost::algorithm::join(arg_types, ", ") + "]";
  } else {
    return "[" + ExtensionFunctionsWhitelist::toString(output_args_) + "]";
  }
}

const std::string TableFunction::getArgNames(bool use_input_args) const {
  std::vector<std::string> names;
  if (use_input_args) {
    for (size_t idx = 0; idx < sql_args_.size(); idx++) {
      const std::vector<std::string> cursor_fields = getCursorFields(idx);
      if (cursor_fields.empty()) {
        const std::string& name = getInputAnnotation(idx, "name", "''");
        names.emplace_back(name);
      } else {
        names.emplace_back("Cursor<" + boost::algorithm::join(cursor_fields, ", ") + ">");
      }
    }
  } else {
    for (size_t idx = 0; idx < output_args_.size(); idx++) {
      const std::string& name = getOutputAnnotation(idx, "name", "''");
      names.emplace_back(name);
    }
  }

  return "[" + boost::algorithm::join(names, ", ") + "]";
}

std::pair<int32_t, int32_t> TableFunction::getInputID(const size_t idx) const {
  // if the annotation is of the form args<INT,INT>, it is refering to a column list
#define PREFIX_LENGTH 5
  const auto& annotation = getOutputAnnotations(idx);
  auto annot = annotation.find("input_id");
  if (annot == annotation.end()) {
    size_t lo = 0;
    for (const auto& ext_arg : input_args_) {
      switch (ext_arg) {
        case ExtArgumentType::TextEncodingDict:
        case ExtArgumentType::ColumnTextEncodingDict:
        case ExtArgumentType::ColumnListTextEncodingDict:
          return std::make_pair(lo, 0);
        default:
          lo++;
      }
    }
    UNREACHABLE();
  }

  const std::string& input_id = annot->second;

  if (input_id == "args<-1>") {
    // empty input id! -1 seems to be the magic number used in RelAlgExecutor.cpp
    return {-1, -1};
  }

  size_t comma = input_id.find(",");
  int32_t gt = input_id.size() - 1;
  int32_t lo = std::stoi(input_id.substr(PREFIX_LENGTH, comma - 1));

  if (comma == std::string::npos) {
    return std::make_pair(lo, 0);
  }
  int32_t hi = std::stoi(input_id.substr(comma + 1, gt - comma - 1));
  return std::make_pair(lo, hi);
}

size_t TableFunction::getSqlOutputRowSizeParameter() const {
  /*
    This function differs from getOutputRowSizeParameter() since it returns the correct
    index for the sizer in the sql_args list. For instance, consider the example below:

      RowMultiplier=4
      input_args=[{i32*, i64}, {i32*, i64}, {i32*, i64}, i32, {i32*, i64}, {i32*, i64},
    i32] sql_args=[cursor, i32, cursor, i32]

    Non-scalar args are aggregated in a cursor inside the sql_args list and the new
    sizer index is 2 rather than 4 originally specified.
  */

  if (hasUserSpecifiedOutputSizeMultiplier()) {
    size_t sizer = getOutputRowSizeParameter();  // lookup until reach the sizer arg
    int32_t ext_arg_index = 0, sql_arg_index = 0;

    auto same_kind = [&](const ExtArgumentType& ext_arg, const ExtArgumentType& sql_arg) {
      return ((is_ext_arg_type_scalar(ext_arg) && is_ext_arg_type_scalar(sql_arg)) ||
              (is_ext_arg_type_nonscalar(ext_arg) && is_ext_arg_type_nonscalar(sql_arg)));
    };

    while ((size_t)ext_arg_index < sizer) {
      if ((size_t)ext_arg_index == sizer - 1)
        return sql_arg_index;

      const auto& ext_arg = input_args_[ext_arg_index];
      const auto& sql_arg = sql_args_[sql_arg_index];

      if (same_kind(ext_arg, sql_arg)) {
        ++ext_arg_index;
        ++sql_arg_index;
      } else {
        CHECK(same_kind(ext_arg, sql_args_[sql_arg_index - 1]));
        ext_arg_index += 1;
      }
    }

    CHECK(false);
  }

  return getOutputRowSizeParameter();
}

bool is_table_function_whitelisted(const std::string& function_name) {
  // All table functions that will be on by default (and not just for testing)
  // must be added to the whitelisted_table_functions set below.
  static const std::unordered_set<std::string> whitelisted_table_functions = {
      "generate_series",
      "generate_random_strings",
      "tf_mandelbrot",
      "tf_mandelbrot_float",
      "tf_mandelbrot_cuda",
      "tf_mandelbrot_cuda_float",
      "tf_geo_rasterize",
      "tf_geo_rasterize_slope",
      "tf_rf_prop",
      "tf_rf_prop_max_signal",
      "tf_point_cloud_metadata",
      "tf_load_point_cloud"};

  return (whitelisted_table_functions.find(function_name) !=
          whitelisted_table_functions.end());
}

void TableFunctionsFactory::add(
    const std::string& name,
    const TableFunctionOutputRowSizer sizer,
    const std::vector<ExtArgumentType>& input_args,
    const std::vector<ExtArgumentType>& output_args,
    const std::vector<ExtArgumentType>& sql_args,
    const std::vector<std::map<std::string, std::string>>& annotations,
    bool is_runtime) {
  static const std::map<std::string, std::string> empty = {};

  auto func_annotations =
      (annotations.size() == sql_args.size() + output_args.size() + 1 ? annotations.back()
                                                                      : empty);
  auto mgr_annotation = func_annotations.find("uses_manager");
  bool uses_manager = mgr_annotation != func_annotations.end() &&
                      boost::algorithm::to_lower_copy(mgr_annotation->second) == "true";

  auto tf = TableFunction(name,
                          sizer,
                          input_args,
                          output_args,
                          sql_args,
                          annotations,
                          is_runtime,
                          uses_manager);
  if (!g_enable_dev_table_functions && !is_runtime &&
      !is_table_function_whitelisted(
          tf.getName(true /* drop_suffix */, true /* lower */))) {
    return;
  }
  auto sig = tf.getSignature(/* include_name */ true, /* include_output */ false);
  for (auto it = functions_.begin(); it != functions_.end();) {
    if (it->second.getName() == name) {
      if (it->second.isRuntime()) {
        LOG(WARNING)
            << "Overriding existing run-time table function (reset not called?): "
            << name;
        it = functions_.erase(it);
      } else {
        throw std::runtime_error("Will not override existing load-time table function: " +
                                 name);
      }
    } else {
      if (sig == it->second.getSignature(/* include_name */ true,
                                         /* include_output */ false) &&
          ((tf.isCPU() && it->second.isCPU()) || (tf.isGPU() && it->second.isGPU()))) {
        LOG(WARNING)
            << "The existing (1) and added (2) table functions have the same signature `"
            << sig << "`:\n"
            << "  1: " << it->second.toString() << "\n  2: " << tf.toString() << "\n";
      }
      ++it;
    }
  }

  functions_.emplace(name, tf);
  if (sizer.type == OutputBufferSizeType::kUserSpecifiedRowMultiplier) {
    auto input_args2 = input_args;
    input_args2.erase(input_args2.begin() + sizer.val - 1);

    auto sql_args2 = sql_args;
    auto sql_sizer_pos = tf.getSqlOutputRowSizeParameter();
    sql_args2.erase(sql_args2.begin() + sql_sizer_pos);

    auto tf2 = TableFunction(name + DEFAULT_ROW_MULTIPLIER_SUFFIX,
                             sizer,
                             input_args2,
                             output_args,
                             sql_args2,
                             annotations,
                             is_runtime,
                             uses_manager);
    auto sig = tf2.getSignature(/* include_name */ true, /* include_output */ false);
    for (auto it = functions_.begin(); it != functions_.end();) {
      if (sig == it->second.getSignature(/* include_name */ true,
                                         /* include_output */ false) &&
          ((tf2.isCPU() && it->second.isCPU()) || (tf2.isGPU() && it->second.isGPU()))) {
        LOG(WARNING)
            << "The existing (1) and added (2) table functions have the same signature `"
            << sig << "`:\n"
            << "  1: " << it->second.toString() << "\n  2: " << tf2.toString() << "\n";
      }
      ++it;
    }
    functions_.emplace(name + DEFAULT_ROW_MULTIPLIER_SUFFIX, tf2);
  }
}

/*
  The implementation for `void TableFunctionsFactory::init()` is
  generated by QueryEngine/scripts/generate_TableFunctionsFactory_init.py
*/

// removes existing runtime table functions
void TableFunctionsFactory::reset() {
  if (!g_enable_table_functions) {
    return;
  }
  for (auto it = functions_.begin(); it != functions_.end();) {
    if (it->second.isRuntime()) {
      it = functions_.erase(it);
    } else {
      ++it;
    }
  }
}

namespace {

std::string drop_suffix_impl(const std::string& str) {
  const auto idx = str.find("__");
  if (idx == std::string::npos) {
    return str;
  }
  CHECK_GT(idx, std::string::size_type(0));
  return str.substr(0, idx);
}

}  // namespace

std::string TableFunction::getName(const bool drop_suffix, const bool lower) const {
  std::string result = name_;
  if (drop_suffix) {
    result = drop_suffix_impl(result);
  }
  if (lower) {
    boost::algorithm::to_lower(result);
  }
  return result;
}

std::string TableFunction::getSignature(const bool include_name,
                                        const bool include_output) const {
  std::string sig;
  if (include_name) {
    sig += getName(/*drop_suffix=*/true, /*lower=*/true);
  }

  size_t arg_idx = 0;
  std::vector<std::string> args;
  for (size_t sql_idx = 0; sql_idx < sql_args_.size(); sql_idx++) {
    const std::vector<std::string> cursor_fields = getCursorFields(sql_idx);
    if (cursor_fields.empty()) {
      const auto& type = ExtensionFunctionsWhitelist::toString(input_args_[arg_idx++]);
      const auto& name = getInputAnnotation(sql_idx, "name", "");
      args.emplace_back(name.empty() ? type : (type + " " + name));
    } else {
      std::vector<std::string> vec;
      for (size_t i = 0; i < cursor_fields.size(); i++) {
        const auto& type = ExtensionFunctionsWhitelist::toString(input_args_[arg_idx++]);
        const auto& name = cursor_fields[i];
        vec.emplace_back((name.empty() ? type : type + " " + name));
      }
      args.emplace_back("Cursor<" + boost::algorithm::join(vec, ", ") + ">");
    }
  }
  sig += "(" + boost::algorithm::join(args, ", ") + ")";
  if (include_output) {
    sig += " -> " + ExtensionFunctionsWhitelist::toString(output_args_);
  }
  return sig;
}

std::string TableFunction::getPreFlightFnName() const {
  // gets the name of the pre flight function associated with this table function
  return getName(false, true) + PREFLIGHT_SUFFIX;
}

std::vector<TableFunction> TableFunctionsFactory::get_table_funcs(const std::string& name,
                                                                  const bool is_gpu) {
  std::vector<TableFunction> table_funcs;
  for (const auto& tf : get_table_funcs(name)) {
    if (is_gpu ? tf.isGPU() : tf.isCPU()) {
      table_funcs.emplace_back(tf);
    }
  }
  return table_funcs;
}

std::vector<TableFunction> TableFunctionsFactory::get_table_funcs(
    const std::string& name) {
  std::vector<TableFunction> table_funcs;
  auto table_func_name = name;
  boost::algorithm::to_lower(table_func_name);
  for (const auto& pair : functions_) {
    auto fname = drop_suffix_impl(pair.first);
    if (fname == table_func_name) {
      table_funcs.push_back(pair.second);
    }
  }
  return table_funcs;
}

std::vector<TableFunction> TableFunctionsFactory::get_table_funcs(const bool is_runtime) {
  std::vector<TableFunction> table_funcs;
  for (const auto& pair : functions_) {
    if (pair.second.isRuntime() == is_runtime) {
      table_funcs.push_back(pair.second);
    }
  }
  return table_funcs;
}

std::vector<TableFunction> TableFunctionsFactory::get_table_funcs() {
  std::vector<TableFunction> table_funcs;
  for (const auto& pair : functions_) {
    table_funcs.push_back(pair.second);
  }
  return table_funcs;
}

std::unordered_map<std::string, TableFunction> TableFunctionsFactory::functions_;

}  // namespace table_functions
