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

#include "Execute.h"

#include "../Shared/sqldefs.h"
#include "Parser/ParserNode.h"

extern "C" uint64_t string_decode(int8_t* chunk_iter_, int64_t pos) {
  auto chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  VarlenDatum vd;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, pos, false, &vd, &is_end);
  CHECK(!is_end);
  return vd.is_null
             ? 0
             : (reinterpret_cast<uint64_t>(vd.pointer) & 0xffffffffffff) | (static_cast<uint64_t>(vd.length) << 48);
}

extern "C" uint64_t string_decompress(const int32_t string_id, const int64_t string_dict_handle) {
  if (string_id == NULL_INT) {
    return 0;
  }
  auto string_dict_proxy = reinterpret_cast<const StringDictionaryProxy*>(string_dict_handle);
  auto string_bytes = string_dict_proxy->getStringBytes(string_id);
  CHECK(string_bytes.first);
  return (reinterpret_cast<uint64_t>(string_bytes.first) & 0xffffffffffff) |
         (static_cast<uint64_t>(string_bytes.second) << 48);
}

extern "C" int32_t string_compress(const int64_t ptr_and_len, const int64_t string_dict_handle) {
  std::string raw_str(reinterpret_cast<char*>(extract_str_ptr_noinline(ptr_and_len)),
                      extract_str_len_noinline(ptr_and_len));
  auto string_dict_proxy = reinterpret_cast<const StringDictionaryProxy*>(string_dict_handle);
  return string_dict_proxy->getIdOfString(raw_str);
}

llvm::Value* Executor::codegen(const Analyzer::CharLengthExpr* expr, const CompilationOptions& co) {
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    if (g_enable_watchdog) {
      throw WatchdogException("LENGTH / CHAR_LENGTH on dictionary-encoded strings would be slow");
    }
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    if (co.device_type_ == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
  }
  std::vector<llvm::Value*> charlength_args{str_lv[1], str_lv[2]};
  std::string fn_name("char_length");
  if (expr->get_calc_encoded_length())
    fn_name += "_encoded";
  const bool is_nullable{!expr->get_arg()->get_type_info().get_notnull()};
  if (is_nullable) {
    fn_name += "_nullable";
    charlength_args.push_back(inlineIntNull(expr->get_type_info()));
  }
  return expr->get_calc_encoded_length()
             ? cgen_state_->emitExternalCall(fn_name, get_int_type(32, cgen_state_->context_), charlength_args)
             : cgen_state_->emitCall(fn_name, charlength_args);
}

llvm::Value* Executor::codegen(const Analyzer::LikeExpr* expr, const CompilationOptions& co) {
  if (is_unnest(extract_cast_arg(expr->get_arg()))) {
    throw std::runtime_error("LIKE not supported for unnested expressions");
  }
  char escape_char{'\\'};
  if (expr->get_escape_expr()) {
    auto escape_char_expr = dynamic_cast<const Analyzer::Constant*>(expr->get_escape_expr());
    CHECK(escape_char_expr);
    CHECK(escape_char_expr->get_type_info().is_string());
    CHECK_EQ(size_t(1), escape_char_expr->get_constval().stringval->size());
    escape_char = (*escape_char_expr->get_constval().stringval)[0];
  }
  auto pattern = dynamic_cast<const Analyzer::Constant*>(expr->get_like_expr());
  CHECK(pattern);
  auto fast_dict_like_lv =
      codegenDictLike(expr->get_own_arg(), pattern, expr->get_is_ilike(), expr->get_is_simple(), escape_char, co);
  if (fast_dict_like_lv) {
    return fast_dict_like_lv;
  }
  const auto& ti = expr->get_arg()->get_type_info();
  CHECK(ti.is_string());
  if (g_enable_watchdog && ti.get_compression() != kENCODING_NONE) {
    throw WatchdogException("Cannot do LIKE / ILIKE on this dictionary encoded column, its cardinality is too high");
  }
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    if (co.device_type_ == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
  }
  auto like_expr_arg_lvs = codegen(expr->get_like_expr(), true, co);
  CHECK_EQ(size_t(3), like_expr_arg_lvs.size());
  const bool is_nullable{!expr->get_arg()->get_type_info().get_notnull()};
  std::vector<llvm::Value*> str_like_args{str_lv[1], str_lv[2], like_expr_arg_lvs[1], like_expr_arg_lvs[2]};
  std::string fn_name{expr->get_is_ilike() ? "string_ilike" : "string_like"};
  if (expr->get_is_simple()) {
    fn_name += "_simple";
  } else {
    str_like_args.push_back(ll_int(int8_t(escape_char)));
  }
  if (is_nullable) {
    fn_name += "_nullable";
    str_like_args.push_back(inlineIntNull(expr->get_type_info()));
  }
  return cgen_state_->emitCall(fn_name, str_like_args);
}

llvm::Value* Executor::codegenDictLike(const std::shared_ptr<Analyzer::Expr> like_arg,
                                       const Analyzer::Constant* pattern,
                                       const bool ilike,
                                       const bool is_simple,
                                       const char escape_char,
                                       const CompilationOptions& co) {
  const auto cast_oper = std::dynamic_pointer_cast<Analyzer::UOper>(like_arg);
  if (!cast_oper) {
    return nullptr;
  }
  CHECK(cast_oper);
  CHECK_EQ(kCAST, cast_oper->get_optype());
  const auto dict_like_arg = cast_oper->get_own_operand();
  const auto& dict_like_arg_ti = dict_like_arg->get_type_info();
  CHECK(dict_like_arg_ti.is_string());
  CHECK_EQ(kENCODING_DICT, dict_like_arg_ti.get_compression());
  const auto sdp = getStringDictionaryProxy(dict_like_arg_ti.get_comp_param(), row_set_mem_owner_, true);
  if (sdp->storageEntryCount() > 200000000) {
    return nullptr;
  }
  const auto& pattern_ti = pattern->get_type_info();
  CHECK(pattern_ti.is_string());
  CHECK_EQ(kENCODING_NONE, pattern_ti.get_compression());
  const auto& pattern_datum = pattern->get_constval();
  const auto& pattern_str = *pattern_datum.stringval;
  const auto matching_ids = sdp->getLike(pattern_str, ilike, is_simple, escape_char);
  // InIntegerSet requires 64-bit values
  std::vector<int64_t> matching_ids_64(matching_ids.size());
  std::copy(matching_ids.begin(), matching_ids.end(), matching_ids_64.begin());
  const auto in_values =
      std::make_shared<Analyzer::InIntegerSet>(dict_like_arg, matching_ids_64, dict_like_arg_ti.get_notnull());
  return codegen(in_values.get(), co);
}

namespace {

std::vector<int32_t> get_compared_ids(const StringDictionaryProxy* dict,
                                      const SQLOps compare_operator,
                                      const std::string& pattern) {
  std::vector<int> ret;
  switch (compare_operator) {
    case kLT:
      ret = dict->getCompare(pattern, "<");
      break;
    case kLE:
      ret = dict->getCompare(pattern, "<=");
      break;
    case kEQ:
    case kBW_EQ:
      ret = dict->getCompare(pattern, "=");
      break;
    case kGT:
      ret = dict->getCompare(pattern, ">");
      break;
    case kGE:
      ret = dict->getCompare(pattern, ">=");
      break;
    case kNE:
      ret = dict->getCompare(pattern, "<>");
      break;
    default:
      std::runtime_error("unsuported operator for string comparision");
  }
  return ret;
}
}  // namespace

llvm::Value* Executor::codegenDictStrCmp(const std::shared_ptr<Analyzer::Expr> lhs,
                                         const std::shared_ptr<Analyzer::Expr> rhs,
                                         const SQLOps compare_operator,
                                         const CompilationOptions& co) {
  auto rhs_cast_oper = std::dynamic_pointer_cast<const Analyzer::UOper>(rhs);
  auto lhs_cast_oper = std::dynamic_pointer_cast<const Analyzer::UOper>(lhs);
  auto rhs_col_var = std::dynamic_pointer_cast<const Analyzer::ColumnVar>(rhs);
  auto lhs_col_var = std::dynamic_pointer_cast<const Analyzer::ColumnVar>(lhs);
  std::shared_ptr<const Analyzer::UOper> cast_oper;
  std::shared_ptr<const Analyzer::ColumnVar> col_var;
  auto compare_opr = compare_operator;
  if (lhs_col_var && rhs_col_var) {
    if (lhs_col_var->get_type_info().get_comp_param() == rhs_col_var->get_type_info().get_comp_param()) {
      if (compare_operator == kEQ || compare_operator == kNE) {
        // TODO (vraj): implement compare between two dictionary encoded columns which share a dictionary
        return nullptr;
      }
    }
    // TODO (vraj): implement compare between two dictionary encoded columns which don't shared dictionary
    throw std::runtime_error("Decoding two Dictionary encoded columns will be slow");
  } else if (lhs_col_var && rhs_cast_oper) {
    cast_oper.swap(rhs_cast_oper);
    col_var.swap(lhs_col_var);
  } else if (lhs_cast_oper && rhs_col_var) {
    cast_oper.swap(lhs_cast_oper);
    col_var.swap(rhs_col_var);
    switch (compare_operator) {
      case kLT:
        compare_opr = kGT;
        break;
      case kLE:
        compare_opr = kGE;
        break;
      case kGT:
        compare_opr = kLT;
        break;
      case kGE:
        compare_opr = kLE;
      default:
        break;
    }
  }
  if (!cast_oper || !col_var) {
    return nullptr;
  }
  CHECK_EQ(kCAST, cast_oper->get_optype());

  const auto const_expr = dynamic_cast<Analyzer::Constant*>(cast_oper->get_own_operand().get());
  if (!const_expr) {
    // Analyzer casts dictionary encoded columns to none encoded if there is a comparison between two encoded columns.
    // Which we currently do not handle.
    return nullptr;
  }
  const auto& const_val = const_expr->get_constval();

  const auto col_ti = col_var->get_type_info();
  CHECK(col_ti.is_string());
  CHECK_EQ(kENCODING_DICT, col_ti.get_compression());
  const auto sdp = getStringDictionaryProxy(col_ti.get_comp_param(), row_set_mem_owner_, true);

  if (!g_fast_strcmp && sdp->storageEntryCount() > 200000000) {
    std::runtime_error("Cardinality for string dictionary is too high");
    return nullptr;
  }

  const auto& pattern_str = *const_val.stringval;
  const auto matching_ids = get_compared_ids(sdp, compare_opr, pattern_str);

  // InIntegerSet requires 64-bit values
  std::vector<int64_t> matching_ids_64(matching_ids.size());
  std::copy(matching_ids.begin(), matching_ids.end(), matching_ids_64.begin());

  const auto in_values = std::make_shared<Analyzer::InIntegerSet>(col_var, matching_ids_64, col_ti.get_notnull());
  return codegen(in_values.get(), co);
}

llvm::Value* Executor::codegen(const Analyzer::RegexpExpr* expr, const CompilationOptions& co) {
  if (is_unnest(extract_cast_arg(expr->get_arg()))) {
    throw std::runtime_error("REGEXP not supported for unnested expressions");
  }
  if (co.device_type_ == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
  }
  char escape_char{'\\'};
  if (expr->get_escape_expr()) {
    auto escape_char_expr = dynamic_cast<const Analyzer::Constant*>(expr->get_escape_expr());
    CHECK(escape_char_expr);
    CHECK(escape_char_expr->get_type_info().is_string());
    CHECK_EQ(size_t(1), escape_char_expr->get_constval().stringval->size());
    escape_char = (*escape_char_expr->get_constval().stringval)[0];
  }
  auto pattern = dynamic_cast<const Analyzer::Constant*>(expr->get_pattern_expr());
  CHECK(pattern);
  auto fast_dict_pattern_lv = codegenDictRegexp(expr->get_own_arg(), pattern, escape_char, co);
  if (fast_dict_pattern_lv) {
    return fast_dict_pattern_lv;
  }
  const auto& ti = expr->get_arg()->get_type_info();
  CHECK(ti.is_string());
  if (g_enable_watchdog && ti.get_compression() != kENCODING_NONE) {
    throw WatchdogException("Cannot do REGEXP_LIKE on this dictionary encoded column, its cardinality is too high");
  }
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
  }
  auto regexp_expr_arg_lvs = codegen(expr->get_pattern_expr(), true, co);
  CHECK_EQ(size_t(3), regexp_expr_arg_lvs.size());
  const bool is_nullable{!expr->get_arg()->get_type_info().get_notnull()};
  std::vector<llvm::Value*> regexp_args{str_lv[1], str_lv[2], regexp_expr_arg_lvs[1], regexp_expr_arg_lvs[2]};
  std::string fn_name("regexp_like");
  regexp_args.push_back(ll_int(int8_t(escape_char)));
  if (is_nullable) {
    fn_name += "_nullable";
    regexp_args.push_back(inlineIntNull(expr->get_type_info()));
    return cgen_state_->emitExternalCall(fn_name, get_int_type(8, cgen_state_->context_), regexp_args);
  }
  return cgen_state_->emitExternalCall(fn_name, get_int_type(1, cgen_state_->context_), regexp_args);
}

llvm::Value* Executor::codegenDictRegexp(const std::shared_ptr<Analyzer::Expr> pattern_arg,
                                         const Analyzer::Constant* pattern,
                                         const char escape_char,
                                         const CompilationOptions& co) {
  const auto cast_oper = std::dynamic_pointer_cast<Analyzer::UOper>(pattern_arg);
  if (!cast_oper) {
    return nullptr;
  }
  CHECK(cast_oper);
  CHECK_EQ(kCAST, cast_oper->get_optype());
  const auto dict_regexp_arg = cast_oper->get_own_operand();
  const auto& dict_regexp_arg_ti = dict_regexp_arg->get_type_info();
  CHECK(dict_regexp_arg_ti.is_string());
  CHECK_EQ(kENCODING_DICT, dict_regexp_arg_ti.get_compression());
  const auto sdp = getStringDictionaryProxy(dict_regexp_arg_ti.get_comp_param(), row_set_mem_owner_, true);
  if (sdp->storageEntryCount() > 15000000) {
    return nullptr;
  }
  const auto& pattern_ti = pattern->get_type_info();
  CHECK(pattern_ti.is_string());
  CHECK_EQ(kENCODING_NONE, pattern_ti.get_compression());
  const auto& pattern_datum = pattern->get_constval();
  const auto& pattern_str = *pattern_datum.stringval;
  const auto matching_ids = sdp->getRegexpLike(pattern_str, escape_char);
  // InIntegerSet requires 64-bit values
  std::vector<int64_t> matching_ids_64(matching_ids.size());
  std::copy(matching_ids.begin(), matching_ids.end(), matching_ids_64.begin());
  const auto in_values =
      std::make_shared<Analyzer::InIntegerSet>(dict_regexp_arg, matching_ids_64, dict_regexp_arg_ti.get_notnull());
  return codegen(in_values.get(), co);
}
