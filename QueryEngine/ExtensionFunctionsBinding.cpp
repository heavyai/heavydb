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

static int match_arguments(const SQLTypeInfo& arg_type,
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
  auto stype = sig_types[sig_pos];
  int max_pos = sig_types.size() - 1;
  switch (arg_type.get_type()) {
    case kBOOLEAN:
      if (stype == ExtArgumentType::Bool) {
        penalty_score += 1000;
        return 1;
      }
      break;
    case kTINYINT:
      switch (stype) {
        case ExtArgumentType::Int8:
          penalty_score += 1000;
          break;
        case ExtArgumentType::Int16:
          penalty_score += 2000;
          break;
        case ExtArgumentType::Int32:
          penalty_score += 4000;
          break;
        case ExtArgumentType::Int64:
          penalty_score += 8000;
          break;
        case ExtArgumentType::Double:
          penalty_score += 1008000;
          break;  // temporary: allow integers as double arguments
        default:
          return -1;
      }
      return 1;
    case kSMALLINT:
      switch (stype) {
        case ExtArgumentType::Int16:
          penalty_score += 1000;
          break;
        case ExtArgumentType::Int32:
          penalty_score += 2000;
          break;
        case ExtArgumentType::Int64:
          penalty_score += 4000;
          break;
        case ExtArgumentType::Double:
          penalty_score += 1004000;
          break;  // temporary: allow integers as double arguments
        default:
          return -1;
      }
      return 1;
    case kINT:
      switch (stype) {
        case ExtArgumentType::Int32:
          penalty_score += 1000;
          break;
        case ExtArgumentType::Int64:
          penalty_score += 2000;
          break;
        case ExtArgumentType::Double:
          penalty_score += 1002000;
          break;  // temporary: allow integers as double arguments
        default:
          return -1;
      }
      return 1;
    case kBIGINT:
      switch (stype) {
        case ExtArgumentType::Int64:
          penalty_score += 1000;
          break;
        case ExtArgumentType::Double:
          penalty_score += 1001000;
          break;  // temporary: allow integers as double arguments
        default:
          return -1;
      }
      return 1;
    case kFLOAT:
      switch (stype) {
        case ExtArgumentType::Float:
          penalty_score += 1000;
          break;
        case ExtArgumentType::Double:
          penalty_score += 2000;
          break;  // is it ok to use floats as double arguments?
        default:
          return -1;
      }
      return 1;
    case kDOUBLE:
      if (stype == ExtArgumentType::Double) {
        penalty_score += 1000;
        return 1;
      }
      break;

    case kPOINT:
    case kLINESTRING:
      if ((stype == ExtArgumentType::PInt8 || stype == ExtArgumentType::PInt16 ||
           stype == ExtArgumentType::PInt32 || stype == ExtArgumentType::PInt64 ||
           stype == ExtArgumentType::PFloat || stype == ExtArgumentType::PDouble) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      } else if (stype == ExtArgumentType::GeoPoint ||
                 stype == ExtArgumentType::GeoLineString) {
        penalty_score += 1000;
        return 1;
      }
      break;
    case kARRAY:
      if ((stype == ExtArgumentType::PInt8 || stype == ExtArgumentType::PInt16 ||
           stype == ExtArgumentType::PInt32 || stype == ExtArgumentType::PInt64 ||
           stype == ExtArgumentType::PFloat || stype == ExtArgumentType::PDouble ||
           stype == ExtArgumentType::PBool) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      } else if (is_ext_arg_type_array(stype)) {
        // array arguments must match exactly
        CHECK(arg_type.is_array());
        const auto stype_ti = ext_arg_type_to_type_info(get_array_arg_elem_type(stype));
        if (arg_type.get_elem_type() == kBOOLEAN && stype_ti.get_type() == kTINYINT) {
          /* Boolean array has the same low-level structure as Int8 array. */
          penalty_score += 1000;
          return 1;
        } else if (arg_type.get_elem_type().get_type() == stype_ti.get_type()) {
          penalty_score += 1000;
          return 1;
        } else {
          return -1;
        }
      }
      break;
    case kPOLYGON:
      if (stype == ExtArgumentType::PInt8 && sig_pos + 3 < max_pos &&
          sig_types[sig_pos + 1] == ExtArgumentType::Int64 &&
          sig_types[sig_pos + 2] == ExtArgumentType::PInt32 &&
          sig_types[sig_pos + 3] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 4;
      } else if (stype == ExtArgumentType::GeoPolygon) {
        penalty_score += 1000;
        return 1;
      }

      break;

    case kMULTIPOLYGON:
      if (stype == ExtArgumentType::PInt8 && sig_pos + 5 < max_pos &&
          sig_types[sig_pos + 1] == ExtArgumentType::Int64 &&
          sig_types[sig_pos + 2] == ExtArgumentType::PInt32 &&
          sig_types[sig_pos + 3] == ExtArgumentType::Int64 &&
          sig_types[sig_pos + 4] == ExtArgumentType::PInt32 &&
          sig_types[sig_pos + 5] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 6;
      } else if (stype == ExtArgumentType::GeoMultiPolygon) {
        penalty_score += 1000;
        return 1;
      }
      break;
    case kDECIMAL:
    case kNUMERIC:
      if (stype == ExtArgumentType::Double && arg_type.get_logical_size() == 8) {
        penalty_score += 1000;
        return 1;
      }
      if (stype == ExtArgumentType::Float && arg_type.get_logical_size() == 4) {
        penalty_score += 1000;
        return 1;
      }
      break;
    case kNULLT:  // NULL maps to a pointer and size argument
      if ((stype == ExtArgumentType::PInt8 || stype == ExtArgumentType::PInt16 ||
           stype == ExtArgumentType::PInt32 || stype == ExtArgumentType::PInt64 ||
           stype == ExtArgumentType::PFloat || stype == ExtArgumentType::PDouble ||
           stype == ExtArgumentType::PBool) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      }
      break;
    case kCOLUMN:
      if (is_ext_arg_type_column(stype)) {
        // column arguments must match exactly
        const auto stype_ti = ext_arg_type_to_type_info(get_column_arg_elem_type(stype));
        if (arg_type.get_elem_type() == kBOOLEAN && stype_ti.get_type() == kTINYINT) {
          /* Boolean column has the same low-level structure as Int8 column. */
          penalty_score += 1000;
          return 1;
        } else if (arg_type.get_elem_type().get_type() == stype_ti.get_type()) {
          penalty_score += 1000;
          return 1;
        } else {
          return -1;
        }
      }
      break;
    case kTEXT:
      switch (arg_type.get_compression()) {
        case kENCODING_NONE:
          if (stype == ExtArgumentType::TextEncodingNone) {
            penalty_score += 1000;
            return 1;
          }
          return -1;
        case kENCODING_DICT:
          if (stype == ExtArgumentType::TextEncodingDict32) {
            penalty_score += 1000;
            return 1;
          }
        default:;
          // todo: dict(8) and dict(16) encodings
      }
      /* Not implemented types:
         kCHAR
         kVARCHAR
         kTIME
         kTIMESTAMP
         kDATE
         kINTERVAL_DAY_TIME
         kINTERVAL_YEAR_MONTH
         kGEOMETRY
         kGEOGRAPHY
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
T bind_function(std::string name,
                Analyzer::ExpressionPtrVector func_args,
                const std::vector<T>& ext_funcs,
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

       For array and geo types, the matching means that the argument
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

  std::vector<SQLTypeInfo> type_infos;
  for (auto atype : func_args) {
    if constexpr (std::is_same_v<T, table_functions::TableFunction>) {
      if (dynamic_cast<const Analyzer::ColumnVar*>(atype.get())) {
        auto ti = SQLTypeInfo(
            kCOLUMN, 0, 0, false, kENCODING_NONE, 0, atype->get_type_info().get_type());
        type_infos.push_back(ti);
        continue;
      }
    }
    type_infos.push_back(atype->get_type_info());
  }

  for (auto ext_func : ext_funcs) {
    index++;
    auto ext_func_args = ext_func.getInputArgs();
    /* In general, `arg_types.size() <= ext_func_args.size()` because
       non-scalar arguments (such as arrays and geo-objects) are
       mapped to multiple `ext_func` arguments. */
    if (func_args.size() <= ext_func_args.size()) {
      /* argument type must fit into the corresponding signature
         argument type, reject signature if not */
      int penalty_score = 0;
      int pos = 0;
      for (auto ti : type_infos) {
        int offset = match_arguments(ti, pos, ext_func_args, penalty_score);
        if (offset < 0) {
          // atype does not match with ext_func argument
          pos = -1;
          break;
        }
        pos += offset;
      }
      if (pos >= 0) {
        // prefer smaller return types
        penalty_score += ext_arg_type_to_type_info(ext_func.getRet()).get_logical_size();
        if (penalty_score < minimal_score) {
          optimal = index;
          minimal_score = penalty_score;
        }
      }
    }
  }

  if (optimal == -1) {
    /* no extension function found that argument types would match
       with types in `arg_types` */
    auto sarg_types = ExtensionFunctionsWhitelist::toString(type_infos);
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
        message += "\n    " + ext_func.toStringSQL();
      }
    }
    throw ExtensionFunctionBindingError(message);
  }
  return ext_funcs[optimal];
}

const table_functions::TableFunction bind_table_function(
    std::string name,
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
    return bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor);
  } catch (ExtensionFunctionBindingError& e) {
    if (is_gpu) {
      is_gpu = false;
      processor = "GPU|CPU";
      ext_funcs = ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
      return bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor);
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
  return bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor);
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

const table_functions::TableFunction bind_table_function(
    std::string name,
    Analyzer::ExpressionPtrVector input_args,
    const bool is_gpu) {
  // used in RelAlgExecutor.cpp
  std::vector<table_functions::TableFunction> table_funcs =
      table_functions::TableFunctionsFactory::get_table_funcs(name, is_gpu);
  return bind_table_function(name, input_args, table_funcs, is_gpu);
}

bool is_ext_arg_type_array(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::ArrayInt8:
    case ExtArgumentType::ArrayInt16:
    case ExtArgumentType::ArrayInt32:
    case ExtArgumentType::ArrayInt64:
    case ExtArgumentType::ArrayFloat:
    case ExtArgumentType::ArrayDouble:
    case ExtArgumentType::ArrayBool:
      return true;

    default:
      return false;
  }
}

bool is_ext_arg_type_column(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::ColumnInt8:
    case ExtArgumentType::ColumnInt16:
    case ExtArgumentType::ColumnInt32:
    case ExtArgumentType::ColumnInt64:
    case ExtArgumentType::ColumnFloat:
    case ExtArgumentType::ColumnDouble:
    case ExtArgumentType::ColumnBool:
      return true;

    default:
      return false;
  }
}

bool is_ext_arg_type_geo(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::GeoPoint:
    case ExtArgumentType::GeoLineString:
    case ExtArgumentType::GeoPolygon:
    case ExtArgumentType::GeoMultiPolygon:
      return true;

    default:
      return false;
  }
}

bool is_ext_arg_type_pointer(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::PInt8:
    case ExtArgumentType::PInt16:
    case ExtArgumentType::PInt32:
    case ExtArgumentType::PInt64:
    case ExtArgumentType::PFloat:
    case ExtArgumentType::PDouble:
    case ExtArgumentType::PBool:
      return true;

    default:
      return false;
  }
}

bool is_ext_arg_type_scalar(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::Int8:
    case ExtArgumentType::Int16:
    case ExtArgumentType::Int32:
    case ExtArgumentType::Int64:
    case ExtArgumentType::Float:
    case ExtArgumentType::Double:
    case ExtArgumentType::Bool:
      return true;

    default:
      return false;
  }
}
