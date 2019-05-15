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
    case kLINESTRING:
    case kPOINT:
    case kARRAY:
      if ((stype == ExtArgumentType::PInt8 || stype == ExtArgumentType::PInt16 ||
           stype == ExtArgumentType::PInt32 || stype == ExtArgumentType::PInt64 ||
           stype == ExtArgumentType::PFloat || stype == ExtArgumentType::PDouble) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      }
      break;
    case kPOLYGON:
      if (stype == ExtArgumentType::PInt8 && sig_pos + 3 < max_pos &&
          sig_types[sig_pos + 1] == ExtArgumentType::Int64 &&
          sig_types[sig_pos + 2] == ExtArgumentType::PInt32 &&
          sig_types[sig_pos + 3] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 4;
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
           stype == ExtArgumentType::PFloat || stype == ExtArgumentType::PDouble) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      }
      break;
      /* Not implemented types:
         kCHAR
         kVARCHAR
         kTIME
         kTIMESTAMP
         kTEXT
         kDATE
         kINTERVAL_DAY_TIME
         kINTERVAL_YEAR_MONTH
         kGEOMETRY
         kGEOGRAPHY
         kEVAL_CONTEXT_TYPE
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

}  // namespace

ExtensionFunction bind_function(std::string name,
                                Analyzer::ExpressionPtrVector func_args,
                                const std::vector<ExtensionFunction>& ext_funcs) {
  // worker function
  /*
    Return extension function that has the following properties

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
  int minimal_score = std::numeric_limits<int>::max();
  int index = -1;
  int optimal = -1;
  for (auto ext_func : ext_funcs) {
    index++;
    auto ext_func_args = ext_func.getArgs();
    /* In general, `arg_types.size() <= ext_func_args.size()` because
       non-scalar arguments (such as arrays and geo-objects) are
       mapped to multiple `ext_func` arguments. */
    if (func_args.size() <= ext_func_args.size()) {
      /* argument type must fit into the corresponding signature
         argument type, reject signature if not */
      int penalty_score = 0;
      int pos = 0;
      for (auto atype : func_args) {
        int offset =
            match_arguments(atype->get_type_info(), pos, ext_func_args, penalty_score);
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
    std::vector<SQLTypeInfo> arg_types;
    for (size_t i = 0; i < func_args.size(); ++i) {
      arg_types.push_back(func_args[i]->get_type_info());
    }
    auto sarg_types = ExtensionFunctionsWhitelist::toString(arg_types);
    if (!ext_funcs.size()) {
      throw std::runtime_error("Function " + name + "(" + sarg_types +
                               ") not supported.");
    }
    auto choices = ExtensionFunctionsWhitelist::toString(ext_funcs, "    ");
    throw std::runtime_error(
        "Function " + name + "(" + sarg_types +
        ") not supported.\n  Existing extension function implementations:\n" + choices);
  }
  return ext_funcs[optimal];
}

ExtensionFunction bind_function(std::string name,
                                Analyzer::ExpressionPtrVector func_args) {
  // used in RelAlgTranslator.cpp
  std::vector<ExtensionFunction> ext_funcs =
      ExtensionFunctionsWhitelist::get_ext_funcs(name);
  return bind_function(name, func_args, ext_funcs);
}

ExtensionFunction bind_function(const Analyzer::FunctionOper* function_oper) {
  // used in ExtensionIR.cpp
  auto name = function_oper->getName();
  Analyzer::ExpressionPtrVector func_args = {};
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    func_args.push_back(function_oper->getOwnArg(i));
  }
  return bind_function(name, func_args);
}
