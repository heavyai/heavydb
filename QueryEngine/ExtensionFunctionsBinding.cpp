#include "ExtensionFunctionsBinding.h"

#include "../Analyzer/Analyzer.h"

namespace {

unsigned narrowing_conversion_score(const SQLTypeInfo& arg_ti, const SQLTypeInfo& arg_target_ti) {
  CHECK(arg_ti.is_number());
  CHECK(arg_target_ti.is_integer() || arg_target_ti.is_fp());
  if (arg_ti.get_type() == arg_target_ti.get_type()) {
    return 0;
  }
  if (arg_ti.get_type() == kDOUBLE && arg_target_ti.get_type() == kFLOAT) {
    return 1;
  }
  if (arg_ti.is_integer()) {
    if (!arg_target_ti.is_integer() || arg_target_ti.get_logical_size() >= arg_ti.get_logical_size()) {
      return 0;
    }
    CHECK_EQ(0, arg_ti.get_logical_size() % arg_target_ti.get_logical_size());
    const int size_ratio = arg_ti.get_logical_size() / arg_target_ti.get_logical_size();
    switch (size_ratio) {
      case 2:
        return 1;
      case 4:
        return 2;
      default:
        CHECK(false);
    }
  }
  if (arg_target_ti.is_integer()) {
    switch (arg_target_ti.get_type()) {
      case kBIGINT:
        return 1;
      case kINT:
        return 2;
      case kSMALLINT:
        return 3;
      default:
        CHECK(false);
    }
  }
  return 0;
}

unsigned widening_conversion_score(const SQLTypeInfo& arg_ti, const SQLTypeInfo& arg_target_ti) {
  CHECK(arg_ti.is_number());
  CHECK(arg_target_ti.is_integer() || arg_target_ti.is_fp());
  if (arg_ti.get_type() == arg_target_ti.get_type()) {
    return 0;
  }
  if (arg_ti.get_type() == kFLOAT && arg_target_ti.get_type() == kDOUBLE) {
    return 1;
  }
  if (arg_ti.is_integer() && arg_target_ti.is_fp()) {
    switch (arg_target_ti.get_type()) {
      case kFLOAT:
        return 1;
      case kDOUBLE:
        return 2;
      default:
        CHECK(false);
    }
  }
  if (arg_ti.is_integer() && arg_target_ti.is_integer()) {
    if (arg_target_ti.get_logical_size() <= arg_ti.get_logical_size()) {
      return 0;
    }
    CHECK_EQ(0, arg_target_ti.get_logical_size() % arg_ti.get_logical_size());
    const int size_ratio = arg_target_ti.get_logical_size() / arg_ti.get_logical_size();
    switch (size_ratio) {
      case 2:
        return 1;
      case 4:
        return 2;
      default:
        CHECK(false);
    }
    return 1;
  }
  return 0;
}

std::vector<unsigned> compute_narrowing_conv_scores(const Analyzer::FunctionOper* function_oper,
                                                    const std::vector<ExtensionFunction>& ext_func_sigs) {
  std::vector<unsigned> narrowing_conv_scores;
  for (const auto& ext_func_sig : ext_func_sigs) {
    const auto& ext_func_args = ext_func_sig.getArgs();
    unsigned score = 0;
    for (size_t i = 0; i < function_oper->getArity(); ++i) {
      const auto arg = function_oper->getArg(i);
      const auto& arg_ti = arg->get_type_info();
      const auto arg_target_ti = ext_arg_type_to_type_info(ext_func_args[i]);
      score += narrowing_conversion_score(arg_ti, arg_target_ti);
    }
    narrowing_conv_scores.push_back(score);
  }
  CHECK_EQ(narrowing_conv_scores.size(), ext_func_sigs.size());
  return narrowing_conv_scores;
}

std::vector<unsigned> compute_widening_conv_scores(const Analyzer::FunctionOper* function_oper,
                                                   const std::vector<const ExtensionFunction*>& ext_func_sigs) {
  std::vector<unsigned> widening_conv_scores;
  for (const auto& ext_func_sig_ptr : ext_func_sigs) {
    const auto& ext_func_args = ext_func_sig_ptr->getArgs();
    unsigned score = 0;
    for (size_t i = 0; i < function_oper->getArity(); ++i) {
      const auto arg = function_oper->getArg(i);
      const auto& arg_ti = arg->get_type_info();
      const auto arg_target_ti = ext_arg_type_to_type_info(ext_func_args[i]);
      score += widening_conversion_score(arg_ti, arg_target_ti);
    }
    widening_conv_scores.push_back(score);
  }
  CHECK_EQ(widening_conv_scores.size(), ext_func_sigs.size());
  return widening_conv_scores;
}

}  // namespace

SQLTypeInfo ext_arg_type_to_type_info(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::Int16:
      return SQLTypeInfo(kSMALLINT, true);
    case ExtArgumentType::Int32:
      return SQLTypeInfo(kINT, true);
    case ExtArgumentType::Int64:
      return SQLTypeInfo(kBIGINT, true);
    case ExtArgumentType::Float:
      return SQLTypeInfo(kFLOAT, true);
    case ExtArgumentType::Double:
      return SQLTypeInfo(kDOUBLE, true);
    default:
      CHECK(false);
  }
  CHECK(false);
  return SQLTypeInfo(kNULLT, false);
}

const ExtensionFunction& bind_function(const Analyzer::FunctionOper* function_oper,
                                       const std::vector<ExtensionFunction>& ext_func_sigs) {
  CHECK(!ext_func_sigs.empty());
  const auto narrowing_conv_scores = compute_narrowing_conv_scores(function_oper, ext_func_sigs);
  const auto min_narrowing_it = std::min_element(narrowing_conv_scores.begin(), narrowing_conv_scores.end());
  std::vector<const ExtensionFunction*> widening_candidates;
  for (size_t cand_idx = 0; cand_idx < narrowing_conv_scores.size(); ++cand_idx) {
    if (narrowing_conv_scores[cand_idx] == *min_narrowing_it) {
      widening_candidates.push_back(&ext_func_sigs[cand_idx]);
    }
  }
  CHECK(!widening_candidates.empty());
  const auto widening_conv_scores = compute_widening_conv_scores(function_oper, widening_candidates);
  const auto min_widening_it = std::min_element(widening_conv_scores.begin(), widening_conv_scores.end());
  return *widening_candidates[min_widening_it - widening_conv_scores.begin()];
}
