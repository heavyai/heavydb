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

#include "ExtensionFunctionsBinding.h"
#include <algorithm>
#include "ExternalExecutor.h"

// A rather crude function binding logic based on the types of the arguments.
// We want it to be possible to write specialized versions of functions to be
// exposed as SQL extensions. This is important especially for performance
// reasons, since double operations can be significantly slower than float. We
// compute a score for each candidate signature based on conversions required to
// from the function arguments as specified in the SQL query to the versions in
// ExtensionFunctions.hpp.

/*
  New implementation for binding a SQL function operator to the
  optimal candidate within in all available extension functions.
 */

namespace {

ExtArgumentType get_column_arg_elem_type(const ExtArgumentType ext_arg_column_type) {
  switch (ext_arg_column_type) {
    case ExtArgumentType::ColumnInt8:
      return ExtArgumentType::Int8;
    case ExtArgumentType::ColumnInt16:
      return ExtArgumentType::Int16;
    case ExtArgumentType::ColumnInt32:
      return ExtArgumentType::Int32;
    case ExtArgumentType::ColumnInt64:
      return ExtArgumentType::Int64;
    case ExtArgumentType::ColumnFloat:
      return ExtArgumentType::Float;
    case ExtArgumentType::ColumnDouble:
      return ExtArgumentType::Double;
    case ExtArgumentType::ColumnBool:
      return ExtArgumentType::Bool;
    case ExtArgumentType::ColumnTextEncodingDict:
      return ExtArgumentType::TextEncodingDict;
    default:
      UNREACHABLE();
  }
  return ExtArgumentType{};
}

ExtArgumentType get_column_list_arg_elem_type(
    const ExtArgumentType ext_arg_column_list_type) {
  switch (ext_arg_column_list_type) {
    case ExtArgumentType::ColumnListInt8:
      return ExtArgumentType::Int8;
    case ExtArgumentType::ColumnListInt16:
      return ExtArgumentType::Int16;
    case ExtArgumentType::ColumnListInt32:
      return ExtArgumentType::Int32;
    case ExtArgumentType::ColumnListInt64:
      return ExtArgumentType::Int64;
    case ExtArgumentType::ColumnListFloat:
      return ExtArgumentType::Float;
    case ExtArgumentType::ColumnListDouble:
      return ExtArgumentType::Double;
    case ExtArgumentType::ColumnListBool:
      return ExtArgumentType::Bool;
    case ExtArgumentType::ColumnListTextEncodingDict:
      return ExtArgumentType::TextEncodingDict;
    default:
      UNREACHABLE();
  }
  return ExtArgumentType{};
}

ExtArgumentType get_array_arg_elem_type(const ExtArgumentType ext_arg_array_type) {
  switch (ext_arg_array_type) {
    case ExtArgumentType::ArrayInt8:
      return ExtArgumentType::Int8;
    case ExtArgumentType::ArrayInt16:
      return ExtArgumentType::Int16;
    case ExtArgumentType::ArrayInt32:
      return ExtArgumentType::Int32;
    case ExtArgumentType::ArrayInt64:
      return ExtArgumentType::Int64;
    case ExtArgumentType::ArrayFloat:
      return ExtArgumentType::Float;
    case ExtArgumentType::ArrayDouble:
      return ExtArgumentType::Double;
    case ExtArgumentType::ArrayBool:
      return ExtArgumentType::Bool;
    default:
      UNREACHABLE();
  }
  return ExtArgumentType{};
}

static int match_numeric_argument(const SQLTypeInfo& arg_type_info,
                                  const bool is_arg_literal,
                                  const ExtArgumentType& sig_ext_arg_type,
                                  int32_t& penalty_score) {
  const auto arg_type = arg_type_info.get_type();
  CHECK(arg_type == kBOOLEAN || arg_type == kTINYINT || arg_type == kSMALLINT ||
        arg_type == kINT || arg_type == kBIGINT || arg_type == kFLOAT ||
        arg_type == kDOUBLE || arg_type == kDECIMAL || arg_type == kNUMERIC);
  // Todo (todd): Add support for timestamp, date, and time types
  const auto sig_type_info = ext_arg_type_to_type_info(sig_ext_arg_type);
  const auto sig_type = sig_type_info.get_type();

  // If we can't legally auto-cast to sig_type, abort
  if (!arg_type_info.is_numeric_scalar_auto_castable(sig_type_info)) {
    return -1;
  }

  // We now compare a measure of the scale of the sig_type with the arg_type,
  // which provides a basis for scoring the match between the two.
  // Note that get_numeric_scalar_scale for the most part returns the logical
  // byte width of the type, with a few caveats for decimals and timestamps
  // described in more depth in comments in the function itself.
  // Also even though for example float and int types return 4 (as in 4 bytes),
  // and double and bigint types return 8, a fp32 type cannot express every 32-bit integer
  // (even if it can cover a larger absolute range), and an fp64 type likewise cannot
  // express every 64-bit integer. However for the purposes of extension function casting
  // we consider floats and ints to be equal scale, and similarly doubles and bigints.
  const auto arg_type_relative_scale = arg_type_info.get_numeric_scalar_scale();
  CHECK_GE(arg_type_relative_scale, 1);
  CHECK_LE(arg_type_relative_scale, 8);
  const auto sig_type_relative_scale = sig_type_info.get_numeric_scalar_scale();
  CHECK_GE(sig_type_relative_scale, 1);
  CHECK_LE(sig_type_relative_scale, 8);

  // Current we do not allow auto-casting to types with less scale/precision, with the
  // caveats around floating point and decimal types mentioned above
  CHECK_GE(sig_type_relative_scale, arg_type_relative_scale);

  // Calculate the ratio of the sig_type by the arg_type, per the above check will be >= 1
  const auto sig_type_scale_gain_ratio =
      sig_type_relative_scale / arg_type_relative_scale;
  CHECK_GE(sig_type_scale_gain_ratio, 1);

  const bool is_integer_to_fp_cast = (arg_type == kTINYINT || arg_type == kSMALLINT ||
                                      arg_type == kINT || arg_type == kBIGINT) &&
                                     (sig_type == kFLOAT || sig_type == kDOUBLE);

  // Following the old bespoke scoring logic this function replaces, we heavily penalize
  // any casts that move ints to floats/doubles for the precision-loss reasons above
  // Arguably all integers in the tinyint and smallint can be fully specified with both
  // float and double types, but we treat them the same as int and bigint types here.
  const int32_t type_family_cast_penalty_score = is_integer_to_fp_cast ? 1001000 : 1000;

  int32_t scale_cast_penalty_score;

  // The following logic is new. Basically there are strong reasons to
  // prefer the promotion of constant literals to the most precise type possible, as
  // rather than the type being inherent in the data - that is a column or columns where
  // a user specified a type (and with any expressions on those columns following our
  // standard sql casting logic), literal types are given to us by Calcite and do not
  // necessarily convey any semantic intent (i.e. 10 will be an int, but 10.0 a decimal)
  // Hence it is better to promote these types to the most precise sig_type available,
  // while at the same time keeping column expressions as close as possible to the input
  // types (mainly for performance, we have many float versions of various functions
  // to allow for greater performance when the underlying data is not of double precision,
  // and hence there is little benefit of the extra cost of computing double precision
  // operators on this data)
  if (is_arg_literal) {
    scale_cast_penalty_score =
        (8000 / arg_type_relative_scale) - (1000 * sig_type_scale_gain_ratio);
  } else {
    scale_cast_penalty_score = (1000 * sig_type_scale_gain_ratio);
  }

  const auto cast_penalty_score =
      type_family_cast_penalty_score + scale_cast_penalty_score;
  CHECK_GT(cast_penalty_score, 0);
  penalty_score += cast_penalty_score;
  return 1;
}

static int match_arguments(const SQLTypeInfo& arg_type,
                           const bool is_arg_literal,
                           int sig_pos,
                           const std::vector<ExtArgumentType>& sig_types,
                           int& penalty_score) {
  /*
    Returns non-negative integer `offset` if `arg_type` and
    `sig_types[sig_pos:sig_pos + offset]` match.

    The `offset` value can be interpreted as the number of extension
    function arguments that is consumed by the given `arg_type`. For
    instance, for scalar types the offset is always 1, for array
    types the offset is 2: one argument for array pointer value and
    one argument for the array size value, etc.

    Returns -1 when the types of an argument and the corresponding
    extension function argument(s) mismatch, or when downcasting would
    be effective.

    In case of non-negative `offset` result, the function updates
    penalty_score argument as follows:

      add 1000 if arg_type is non-scalar, otherwise:
      add 1000 * sizeof(sig_type) / sizeof(arg_type)
      add 1000000 if type kinds differ (integer vs double, for instance)

   */
  int max_pos = sig_types.size() - 1;
  if (sig_pos > max_pos) {
    return -1;
  }
  auto sig_type = sig_types[sig_pos];
  switch (arg_type.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kFLOAT:
    case kDOUBLE:
    case kDECIMAL:
    case kNUMERIC:
      return match_numeric_argument(arg_type, is_arg_literal, sig_type, penalty_score);
    case kARRAY:
      if ((sig_type == ExtArgumentType::PInt8 || sig_type == ExtArgumentType::PInt16 ||
           sig_type == ExtArgumentType::PInt32 || sig_type == ExtArgumentType::PInt64 ||
           sig_type == ExtArgumentType::PFloat || sig_type == ExtArgumentType::PDouble ||
           sig_type == ExtArgumentType::PBool) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      } else if (is_ext_arg_type_array(sig_type)) {
        // array arguments must match exactly
        CHECK(arg_type.is_array());
        const auto sig_type_ti =
            ext_arg_type_to_type_info(get_array_arg_elem_type(sig_type));
        if (arg_type.get_elem_type() == kBOOLEAN && sig_type_ti.get_type() == kTINYINT) {
          /* Boolean array has the same low-level structure as Int8 array. */
          penalty_score += 1000;
          return 1;
        } else if (arg_type.get_elem_type().get_type() == sig_type_ti.get_type()) {
          penalty_score += 1000;
          return 1;
        } else {
          return -1;
        }
      }
      break;
    case kNULLT:  // NULL maps to a pointer and size argument
      if ((sig_type == ExtArgumentType::PInt8 || sig_type == ExtArgumentType::PInt16 ||
           sig_type == ExtArgumentType::PInt32 || sig_type == ExtArgumentType::PInt64 ||
           sig_type == ExtArgumentType::PFloat || sig_type == ExtArgumentType::PDouble ||
           sig_type == ExtArgumentType::PBool) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      }
      break;
    case kCOLUMN:
      if (is_ext_arg_type_column(sig_type)) {
        // column arguments must match exactly
        const auto sig_type_ti =
            ext_arg_type_to_type_info(get_column_arg_elem_type(sig_type));
        if (arg_type.get_elem_type() == kBOOLEAN && sig_type_ti.get_type() == kTINYINT) {
          /* Boolean column has the same low-level structure as Int8 column. */
          penalty_score += 1000;
          return 1;
        } else if (arg_type.get_elem_type().get_type() == sig_type_ti.get_type()) {
          penalty_score += 1000;
          return 1;
        } else {
          return -1;
        }
      }
      break;
    case kCOLUMN_LIST:
      if (is_ext_arg_type_column_list(sig_type)) {
        // column_list arguments must match exactly
        const auto sig_type_ti =
            ext_arg_type_to_type_info(get_column_list_arg_elem_type(sig_type));
        if (arg_type.get_elem_type() == kBOOLEAN && sig_type_ti.get_type() == kTINYINT) {
          /* Boolean column_list has the same low-level structure as Int8 column_list. */
          penalty_score += 10000;
          return 1;
        } else if (arg_type.get_elem_type().get_type() == sig_type_ti.get_type()) {
          penalty_score += 10000;
          return 1;
        } else {
          return -1;
        }
      }
      break;
    case kVARCHAR:
      if (sig_type != ExtArgumentType::TextEncodingNone) {
        return -1;
      }
      switch (arg_type.get_compression()) {
        case kENCODING_NONE:
          penalty_score += 1000;
          return 1;
        case kENCODING_DICT:
          return -1;
          // Todo (todd): Evaluate when and where we can tranlate to dictionary-encoded
        default:
          UNREACHABLE();
      }
    case kTEXT:
      if (sig_type != ExtArgumentType::TextEncodingNone) {
        return -1;
      }
      switch (arg_type.get_compression()) {
        case kENCODING_NONE:
          penalty_score += 1000;
          return 1;
        case kENCODING_DICT:
          return -1;
        default:
          UNREACHABLE();
      }
      /* Not implemented types:
         kCHAR
         kTIME
         kTIMESTAMP
         kDATE
         kINTERVAL_DAY_TIME
         kINTERVAL_YEAR_MONTH
         kEVAL_CONTEXT_TYPE
         kVOID
         kCURSOR
      */
    default:
      throw std::runtime_error(std::string(__FILE__) + "#" + std::to_string(__LINE__) +
                               ": support for " + arg_type.get_type_name() +
                               "(type=" + std::to_string(arg_type.get_type()) + ")" +
                               +" not implemented: \n  pos=" + std::to_string(sig_pos) +
                               " max_pos=" + std::to_string(max_pos) + "\n  sig_types=(" +
                               ExtensionFunctionsWhitelist::toString(sig_types) + ")");
  }
  return -1;
}

bool is_valid_identifier(std::string str) {
  if (!str.size()) {
    return false;
  }

  if (!(std::isalpha(str[0]) || str[0] == '_')) {
    return false;
  }

  for (size_t i = 1; i < str.size(); i++) {
    if (!(std::isalnum(str[i]) || str[i] == '_')) {
      return false;
    }
  }

  return true;
}

}  // namespace

template <typename T>
std::tuple<T, std::vector<SQLTypeInfo>> bind_function(
    std::string name,
    Analyzer::ExpressionPtrVector func_args,  // function args from sql query
    const std::vector<T>& ext_funcs,          // list of functions registered
    const std::string processor) {
  /* worker function

     Template type T must implement the following methods:

       std::vector<ExtArgumentType> getInputArgs()
   */
  /*
    Return extension function/table function that has the following
    properties

    1. each argument type in `arg_types` matches with extension
       function argument types.

       For scalar types, the matching means that the types are either
       equal or the argument type is smaller than the corresponding
       the extension function argument type. This ensures that no
       information is lost when casting of argument values is
       required.

       For array, the matching means that the argument
       type matches exactly with a group of extension function
       argument types. See `match_arguments`.

    2. has minimal penalty score among all implementations of the
       extension function with given `name`, see `get_penalty_score`
       for the definition of penalty score.

    It is assumed that function_oper and extension functions in
    ext_funcs have the same name.
   */
  if (!is_valid_identifier(name)) {
    throw NativeExecutionError(
        "Cannot bind function with invalid UDF/UDTF function name: " + name);
  }

  int minimal_score = std::numeric_limits<int>::max();
  int index = -1;
  int optimal = -1;
  int optimal_variant = -1;

  std::vector<SQLTypeInfo> type_infos_input;
  std::vector<bool> args_are_constants;
  for (auto atype : func_args) {
    if constexpr (std::is_same_v<T, table_functions::TableFunction>) {
      if (dynamic_cast<const Analyzer::ColumnVar*>(atype.get())) {
        SQLTypeInfo type_info = atype->get_type_info();
        if (atype->get_type_info().get_type() == kTEXT) {
          auto ti = generate_column_type(type_info.get_type(),         // subtype
                                         type_info.get_compression(),  // compression
                                         type_info.get_comp_param());  // comp_param
          type_infos_input.push_back(ti);
          args_are_constants.push_back(false);
        } else {
          auto ti = generate_column_type(type_info.get_type());
          type_infos_input.push_back(ti);
          args_are_constants.push_back(true);
        }
        continue;
      }
    }
    type_infos_input.push_back(atype->get_type_info());
    if (dynamic_cast<const Analyzer::Constant*>(atype.get())) {
      args_are_constants.push_back(true);
    } else {
      args_are_constants.push_back(false);
    }
  }
  CHECK_EQ(type_infos_input.size(), args_are_constants.size());

  if (type_infos_input.size() == 0 && ext_funcs.size() > 0) {
    CHECK_EQ(ext_funcs.size(), static_cast<size_t>(1));
    CHECK_EQ(ext_funcs[0].getInputArgs().size(), static_cast<size_t>(0));
    if constexpr (std::is_same_v<T, table_functions::TableFunction>) {
      CHECK(ext_funcs[0].hasNonUserSpecifiedOutputSize());
    }
    std::vector<SQLTypeInfo> empty_type_info_variant(0);
    return {ext_funcs[0], empty_type_info_variant};
  }

  // clang-format off
  /*
    Table functions may have arguments such as ColumnList that collect
    neighboring columns with the same data type into a single object.
    Here we compute all possible combinations of mapping a subset of
    columns into columns sets. For example, if the types of function
    arguments are (as given in func_args argument)

      (Column<int>, Column<int>, Column<int>, int)

    then the computed variants will be

      (Column<int>, Column<int>, Column<int>, int)
      (Column<int>, Column<int>, ColumnList[1]<int>, int)
      (Column<int>, ColumnList[1]<int>, Column<int>, int)
      (Column<int>, ColumnList[2]<int>, int)
      (ColumnList[1]<int>, Column<int>, Column<int>, int)
      (ColumnList[1]<int>, Column<int>, ColumnList[1]<int>, int)
      (ColumnList[2]<int>, Column<int>, int)
      (ColumnList[3]<int>, int)

    where the integers in [..] indicate the number of collected
    columns. In the SQLTypeInfo instance, this number is stored in the
    SQLTypeInfo dimension attribute.

    As an example, let us consider a SQL query containing the
    following expression calling a UDTF foo:

      table(foo(cursor(select a, b, c from tableofints), 1))

    Here follows a list of table functions and the corresponding
    optimal argument type variants that are computed for the given
    query expression:

    UDTF:  foo(ColumnList<int>, RowMultiplier) -> Column<int>
           (ColumnList[3]<int>, int)               # a, b, c are all collected to column_list

    UDTF:  foo(Column<int>, ColumnList<int>, RowMultiplier) -> Column<int>
           (Column<int>, ColumnList[2]<int>, int)  # b and c are collected to column_list

    UDTF:  foo(Column<int>, Column<int>, Column<int>, RowMultiplier) -> Column<int>
           (Column<int>, Column<int>, Column<int>, int)
   */
  // clang-format on
  std::vector<std::vector<SQLTypeInfo>> type_infos_variants;
  for (auto ti : type_infos_input) {
    if (type_infos_variants.begin() == type_infos_variants.end()) {
      type_infos_variants.push_back({ti});
      if constexpr (std::is_same_v<T, table_functions::TableFunction>) {
        if (ti.is_column()) {
          auto mti = generate_column_list_type(ti.get_subtype());
          mti.set_dimension(1);
          type_infos_variants.push_back({mti});
        }
      }
      continue;
    }
    std::vector<std::vector<SQLTypeInfo>> new_type_infos_variants;
    for (auto& type_infos : type_infos_variants) {
      if constexpr (std::is_same_v<T, table_functions::TableFunction>) {
        if (ti.is_column()) {
          auto new_type_infos = type_infos;  // makes a copy
          const auto& last = type_infos.back();
          if (last.is_column_list() && last.get_subtype() == ti.get_subtype()) {
            // last column_list consumes column argument if types match
            new_type_infos.back().set_dimension(last.get_dimension() + 1);
          } else {
            // add column as column_list argument
            auto mti = generate_column_list_type(ti.get_subtype());
            mti.set_dimension(1);
            new_type_infos.push_back(mti);
          }
          new_type_infos_variants.push_back(new_type_infos);
        }
      }
      type_infos.push_back(ti);
    }
    type_infos_variants.insert(type_infos_variants.end(),
                               new_type_infos_variants.begin(),
                               new_type_infos_variants.end());
  }

  // Find extension function that gives the best match on the set of
  // argument type variants:
  for (auto ext_func : ext_funcs) {
    index++;

    auto ext_func_args = ext_func.getInputArgs();
    int index_variant = -1;
    for (const auto& type_infos : type_infos_variants) {
      index_variant++;
      int penalty_score = 0;
      int pos = 0;
      int original_input_idx = 0;
      CHECK_LE(type_infos.size(), args_are_constants.size());
      // for (size_t ti_idx = 0; ti_idx != type_infos.size(); ++ti_idx) {
      for (const auto& ti : type_infos) {
        int offset = match_arguments(ti,
                                     args_are_constants[original_input_idx],
                                     pos,
                                     ext_func_args,
                                     penalty_score);
        if (offset < 0) {
          // atype does not match with ext_func argument
          pos = -1;
          break;
        }
        if (ti.get_type() == kCOLUMN_LIST) {
          original_input_idx += ti.get_dimension();
        } else {
          original_input_idx++;
        }
        pos += offset;
      }

      if ((size_t)pos == ext_func_args.size()) {
        CHECK_EQ(args_are_constants.size(), original_input_idx);
        // prefer smaller return types
        penalty_score += ext_arg_type_to_type_info(ext_func.getRet()).get_logical_size();
        if (penalty_score < minimal_score) {
          optimal = index;
          minimal_score = penalty_score;
          optimal_variant = index_variant;
        }
      }
    }
  }

  if (optimal == -1) {
    /* no extension function found that argument types would match
       with types in `arg_types` */
    auto sarg_types = ExtensionFunctionsWhitelist::toString(type_infos_input);
    std::string message;
    if (!ext_funcs.size()) {
      message = "Function " + name + "(" + sarg_types + ") not supported.";
      throw ExtensionFunctionBindingError(message);
    } else {
      if constexpr (std::is_same_v<T, table_functions::TableFunction>) {
        message = "Could not bind " + name + "(" + sarg_types + ") to any " + processor +
                  " UDTF implementation.";
      } else if constexpr (std::is_same_v<T, ExtensionFunction>) {
        message = "Could not bind " + name + "(" + sarg_types + ") to any " + processor +
                  " UDF implementation.";
      } else {
        LOG(FATAL) << "bind_function: unknown extension function type "
                   << typeid(T).name();
      }
      message += "\n  Existing extension function implementations:";
      for (const auto& ext_func : ext_funcs) {
        // Do not show functions missing the sizer argument
        if constexpr (std::is_same_v<T, table_functions::TableFunction>)
          if (ext_func.useDefaultSizer())
            continue;
        message += "\n    " + ext_func.toStringSQL();
      }
    }
    throw ExtensionFunctionBindingError(message);
  }

  // Functions with "_default_" suffix only exist for calcite
  if constexpr (std::is_same_v<T, table_functions::TableFunction>) {
    if (ext_funcs[optimal].hasUserSpecifiedOutputSizeMultiplier() &&
        ext_funcs[optimal].useDefaultSizer()) {
      std::string name = ext_funcs[optimal].getName();
      name.erase(name.find(DEFAULT_ROW_MULTIPLIER_SUFFIX),
                 sizeof(DEFAULT_ROW_MULTIPLIER_SUFFIX));
      for (size_t i = 0; i < ext_funcs.size(); i++) {
        if (ext_funcs[i].getName() == name) {
          optimal = i;
          std::vector<SQLTypeInfo> type_info = type_infos_variants[optimal_variant];
          size_t sizer = ext_funcs[optimal].getOutputRowSizeParameter();
          type_info.insert(type_info.begin() + sizer - 1, SQLTypeInfo(kINT, true));
          return {ext_funcs[optimal], type_info};
        }
      }
      UNREACHABLE();
    }
  }

  return {ext_funcs[optimal], type_infos_variants[optimal_variant]};
}

const std::tuple<table_functions::TableFunction, std::vector<SQLTypeInfo>>
bind_table_function(std::string name,
                    Analyzer::ExpressionPtrVector input_args,
                    const std::vector<table_functions::TableFunction>& table_funcs,
                    const bool is_gpu) {
  std::string processor = (is_gpu ? "GPU" : "CPU");
  return bind_function<table_functions::TableFunction>(
      name, input_args, table_funcs, processor);
}

ExtensionFunction bind_function(std::string name,
                                Analyzer::ExpressionPtrVector func_args) {
  // used in RelAlgTranslator.cpp, first try GPU UDFs, then fall back
  // to CPU UDFs.
  bool is_gpu = true;
  std::string processor = "GPU";
  auto ext_funcs = ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
  if (!ext_funcs.size()) {
    is_gpu = false;
    processor = "CPU";
    ext_funcs = ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
  }
  try {
    return std::get<0>(
        bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor));
  } catch (ExtensionFunctionBindingError& e) {
    if (is_gpu) {
      is_gpu = false;
      processor = "GPU|CPU";
      ext_funcs = ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
      return std::get<0>(
          bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor));
    } else {
      throw;
    }
  }
}

ExtensionFunction bind_function(std::string name,
                                Analyzer::ExpressionPtrVector func_args,
                                const bool is_gpu) {
  // used below
  std::vector<ExtensionFunction> ext_funcs =
      ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
  std::string processor = (is_gpu ? "GPU" : "CPU");
  return std::get<0>(
      bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor));
}

ExtensionFunction bind_function(const Analyzer::FunctionOper* function_oper,
                                const bool is_gpu) {
  // used in ExtensionsIR.cpp
  auto name = function_oper->getName();
  Analyzer::ExpressionPtrVector func_args = {};
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    func_args.push_back(function_oper->getOwnArg(i));
  }
  return bind_function(name, func_args, is_gpu);
}

const std::tuple<table_functions::TableFunction, std::vector<SQLTypeInfo>>
bind_table_function(std::string name,
                    Analyzer::ExpressionPtrVector input_args,
                    const bool is_gpu) {
  // used in RelAlgExecutor.cpp
  std::vector<table_functions::TableFunction> table_funcs =
      table_functions::TableFunctionsFactory::get_table_funcs(name, is_gpu);
  return bind_table_function(name, input_args, table_funcs, is_gpu);
}
