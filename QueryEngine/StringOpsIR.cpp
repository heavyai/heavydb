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

#include "CodeGenerator.h"
#include "Execute.h"

#include "../Shared/funcannotations.h"
#include "../Shared/sqldefs.h"
#include "Parser/ParserNode.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "StringOps/StringOps.h"

#include <boost/locale/conversion.hpp>

extern "C" RUNTIME_EXPORT uint64_t string_decode(int8_t* chunk_iter_, int64_t pos) {
  auto chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  VarlenDatum vd;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, pos, false, &vd, &is_end);
  CHECK(!is_end);
  return vd.is_null ? 0
                    : (reinterpret_cast<uint64_t>(vd.pointer) & 0xffffffffffff) |
                          (static_cast<uint64_t>(vd.length) << 48);
}

extern "C" RUNTIME_EXPORT uint64_t string_decompress(const int32_t string_id,
                                                     const int64_t string_dict_handle) {
  if (string_id == NULL_INT) {
    return 0;
  }
  auto string_dict_proxy =
      reinterpret_cast<const StringDictionaryProxy*>(string_dict_handle);
  auto string_bytes = string_dict_proxy->getStringBytes(string_id);
  CHECK(string_bytes.first);
  return (reinterpret_cast<uint64_t>(string_bytes.first) & 0xffffffffffff) |
         (static_cast<uint64_t>(string_bytes.second) << 48);
}

extern "C" RUNTIME_EXPORT int32_t string_compress(const int64_t ptr_and_len,
                                                  const int64_t string_dict_handle) {
  std::string raw_str(reinterpret_cast<char*>(extract_str_ptr_noinline(ptr_and_len)),
                      extract_str_len_noinline(ptr_and_len));
  auto string_dict_proxy = reinterpret_cast<StringDictionaryProxy*>(string_dict_handle);
  return string_dict_proxy->getOrAddTransient(raw_str);
}

extern "C" RUNTIME_EXPORT int32_t
apply_string_ops_and_encode(const char* str_ptr,
                            const int32_t str_len,
                            const int64_t string_ops_handle,
                            const int64_t string_dict_handle) {
  std::string raw_str(str_ptr, str_len);
  auto string_ops =
      reinterpret_cast<const StringOps_Namespace::StringOps*>(string_ops_handle);
  auto string_dict_proxy = reinterpret_cast<StringDictionaryProxy*>(string_dict_handle);
  const auto result_str = string_ops->operator()(raw_str);
  if (result_str.empty()) {
    return inline_int_null_value<int32_t>();
  }
  return string_dict_proxy->getOrAddTransient(result_str);
}

extern "C" RUNTIME_EXPORT int32_t
intersect_translate_string_id_to_other_dict(const int32_t string_id,
                                            const int64_t source_string_dict_handle,
                                            const int64_t dest_string_dict_handle) {
  const auto source_string_dict_proxy =
      reinterpret_cast<StringDictionaryProxy*>(source_string_dict_handle);
  auto dest_string_dict_proxy =
      reinterpret_cast<StringDictionaryProxy*>(dest_string_dict_handle);
  // Can we have StringDictionaryProxy::getString return a reference?
  const auto source_str = source_string_dict_proxy->getString(string_id);
  if (source_str.empty()) {
    return inline_int_null_value<int32_t>();
  }
  return dest_string_dict_proxy->getIdOfString(source_str);
}

extern "C" RUNTIME_EXPORT int32_t
union_translate_string_id_to_other_dict(const int32_t string_id,
                                        const int64_t source_string_dict_handle,
                                        const int64_t dest_string_dict_handle) {
  const auto source_string_dict_proxy =
      reinterpret_cast<StringDictionaryProxy*>(source_string_dict_handle);
  auto dest_string_dict_proxy =
      reinterpret_cast<StringDictionaryProxy*>(dest_string_dict_handle);
  // Can we have StringDictionaryProxy::getString return a reference?
  const auto source_str = source_string_dict_proxy->getString(string_id);
  if (source_str.empty()) {
    return inline_int_null_value<int32_t>();
  }
  return dest_string_dict_proxy->getOrAddTransient(source_str);
}

llvm::Value* CodeGenerator::codegen(const Analyzer::CharLengthExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
  }
  std::vector<llvm::Value*> charlength_args{str_lv[1], str_lv[2]};
  std::string fn_name("char_length");
  if (expr->get_calc_encoded_length()) {
    fn_name += "_encoded";
  }
  const bool is_nullable{!expr->get_arg()->get_type_info().get_notnull()};
  if (is_nullable) {
    fn_name += "_nullable";
    charlength_args.push_back(cgen_state_->inlineIntNull(expr->get_type_info()));
  }
  return expr->get_calc_encoded_length()
             ? cgen_state_->emitExternalCall(
                   fn_name, get_int_type(32, cgen_state_->context_), charlength_args)
             : cgen_state_->emitCall(fn_name, charlength_args);
}

llvm::Value* CodeGenerator::codegen(const Analyzer::KeyForStringExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto str_lv = codegen(expr->get_arg(), true, co);
  CHECK_EQ(size_t(1), str_lv.size());
  return cgen_state_->emitCall("key_for_string_encoded", str_lv);
}

std::vector<StringOps_Namespace::StringOpInfo> getStringOpInfos(
    const Analyzer::StringOper* expr) {
  std::vector<StringOps_Namespace::StringOpInfo> string_op_infos;
  auto chained_string_op_exprs = expr->getChainedStringOpExprs();
  if (chained_string_op_exprs.empty()) {
    // Likely will change the below to a CHECK but until we have more confidence
    // that all potential query patterns have nodes that might contain string ops folded,
    // leaving as an error for now
    throw std::runtime_error(
        "Expected folded string operator but found operator unfolded.");
  }
  // Consider encapsulating below in an Analyzer::StringOper method to dedup
  for (const auto& chained_string_op_expr : chained_string_op_exprs) {
    auto chained_string_op =
        dynamic_cast<const Analyzer::StringOper*>(chained_string_op_expr.get());
    CHECK(chained_string_op);
    StringOps_Namespace::StringOpInfo string_op_info(chained_string_op->get_kind(),
                                                     chained_string_op->getLiteralArgs());
    string_op_infos.emplace_back(string_op_info);
  }
  return string_op_infos;
}

llvm::Value* CodeGenerator::codegenPerRowStringOper(const Analyzer::StringOper* expr,
                                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_GE(expr->getArity(), 1UL);

  const auto& expr_ti = expr->get_type_info();
  const auto primary_arg = expr->getArg(0);
  CHECK(primary_arg->get_type_info().is_none_encoded_string());

  if (g_cluster) {
    throw std::runtime_error(
        "Cast from none-encoded string to dictionary-encoded not supported for "
        "distributed queries");
  }
  if (co.device_type == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
  }
  auto primary_str_lv = codegen(primary_arg, true, co);
  CHECK_EQ(size_t(3), primary_str_lv.size());
  const auto string_op_infos = getStringOpInfos(expr);
  CHECK(string_op_infos.size());

  const auto string_ops =
      executor()->getRowSetMemoryOwner()->getStringOps(string_op_infos);
  const int64_t string_ops_handle = reinterpret_cast<int64_t>(string_ops);
  auto string_ops_handle_lv = cgen_state_->llInt(string_ops_handle);

  const int64_t dest_string_proxy_handle =
      reinterpret_cast<int64_t>(executor()->getStringDictionaryProxy(
          expr_ti.get_comp_param(), executor()->getRowSetMemoryOwner(), true));
  auto dest_string_proxy_handle_lv = cgen_state_->llInt(dest_string_proxy_handle);
  std::vector<llvm::Value*> string_oper_lvs{primary_str_lv[1],
                                            primary_str_lv[2],
                                            string_ops_handle_lv,
                                            dest_string_proxy_handle_lv};

  return cgen_state_->emitExternalCall("apply_string_ops_and_encode",
                                       get_int_type(32, cgen_state_->context_),
                                       string_oper_lvs);
}

std::unique_ptr<StringDictionaryTranslationMgr> translate_dict_strings(
    const Analyzer::StringOper* expr,
    const ExecutorDeviceType device_type,
    Executor* executor) {
  const auto& expr_ti = expr->get_type_info();
  const auto dict_id = expr_ti.get_comp_param();
  const auto string_op_infos = getStringOpInfos(expr);
  CHECK(string_op_infos.size());

  auto string_dictionary_translation_mgr =
      std::make_unique<StringDictionaryTranslationMgr>(
          dict_id,
          dict_id,
          false,  // translate_intersection_only
          string_op_infos,
          device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL
                                                 : Data_Namespace::CPU_LEVEL,
          executor->deviceCount(device_type),
          executor,
          &executor->getCatalog()->getDataMgr(),
          false /* delay_translation */);
  return string_dictionary_translation_mgr;
}

llvm::Value* CodeGenerator::codegen(const Analyzer::StringOper* expr,
                                    const CompilationOptions& co) {
  CHECK_GE(expr->getArity(), 1UL);
  if (expr->hasNoneEncodedTextArg()) {
    return codegenPerRowStringOper(expr, co);
  }
  AUTOMATIC_IR_METADATA(cgen_state_);

  const auto& expr_ti = expr->get_type_info();
  auto string_dictionary_translation_mgr =
      translate_dict_strings(expr, co.device_type, executor());

  auto str_id_lv = codegen(expr->getArg(0), true, co);
  CHECK_EQ(size_t(1), str_id_lv.size());

  return cgen_state_
      ->moveStringDictionaryTranslationMgr(std::move(string_dictionary_translation_mgr))
      ->codegen(str_id_lv[0], expr_ti, true /* add_nullcheck */, co);
}

// Method below is for join probes, as we cast the StringOper nodes to ColumnVars early to
// not special case that codepath (but retain the StringOpInfos, which we use here to
// execute the same string ops as we would on a native StringOper node)
llvm::Value* CodeGenerator::codegenPseudoStringOper(
    const Analyzer::ColumnVar* expr,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto& expr_ti = expr->get_type_info();
  const auto dict_id = expr_ti.get_comp_param();

  auto string_dictionary_translation_mgr =
      std::make_unique<StringDictionaryTranslationMgr>(
          dict_id,
          dict_id,
          false,  // translate_intersection_only
          string_op_infos,
          co.device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL
                                                    : Data_Namespace::CPU_LEVEL,
          executor()->deviceCount(co.device_type),
          executor(),
          &executor()->getCatalog()->getDataMgr(),
          false /* delay_translation */);

  auto str_id_lv = codegen(expr, true /* fetch_column */, co);
  CHECK_EQ(size_t(1), str_id_lv.size());

  return cgen_state_
      ->moveStringDictionaryTranslationMgr(std::move(string_dictionary_translation_mgr))
      ->codegen(str_id_lv[0], expr_ti, true /* add_nullcheck */, co);
}

llvm::Value* CodeGenerator::codegen(const Analyzer::LikeExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (is_unnest(extract_cast_arg(expr->get_arg()))) {
    throw std::runtime_error("LIKE not supported for unnested expressions");
  }
  char escape_char{'\\'};
  if (expr->get_escape_expr()) {
    auto escape_char_expr =
        dynamic_cast<const Analyzer::Constant*>(expr->get_escape_expr());
    CHECK(escape_char_expr);
    CHECK(escape_char_expr->get_type_info().is_string());
    CHECK_EQ(size_t(1), escape_char_expr->get_constval().stringval->size());
    escape_char = (*escape_char_expr->get_constval().stringval)[0];
  }
  auto pattern = dynamic_cast<const Analyzer::Constant*>(expr->get_like_expr());
  CHECK(pattern);
  auto fast_dict_like_lv = codegenDictLike(expr->get_own_arg(),
                                           pattern,
                                           expr->get_is_ilike(),
                                           expr->get_is_simple(),
                                           escape_char,
                                           co);
  if (fast_dict_like_lv) {
    return fast_dict_like_lv;
  }
  const auto& ti = expr->get_arg()->get_type_info();
  CHECK(ti.is_string());
  if (g_enable_watchdog && ti.get_compression() != kENCODING_NONE) {
    throw WatchdogException(
        "Cannot do LIKE / ILIKE on this dictionary encoded column, its cardinality is "
        "too high");
  }
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
  }
  auto like_expr_arg_lvs = codegen(expr->get_like_expr(), true, co);
  CHECK_EQ(size_t(3), like_expr_arg_lvs.size());
  const bool is_nullable{!expr->get_arg()->get_type_info().get_notnull()};
  std::vector<llvm::Value*> str_like_args{
      str_lv[1], str_lv[2], like_expr_arg_lvs[1], like_expr_arg_lvs[2]};
  std::string fn_name{expr->get_is_ilike() ? "string_ilike" : "string_like"};
  if (expr->get_is_simple()) {
    fn_name += "_simple";
  } else {
    str_like_args.push_back(cgen_state_->llInt(int8_t(escape_char)));
  }
  if (is_nullable) {
    fn_name += "_nullable";
    str_like_args.push_back(cgen_state_->inlineIntNull(expr->get_type_info()));
  }
  return cgen_state_->emitCall(fn_name, str_like_args);
}

void pre_translate_string_ops(const Analyzer::StringOper* string_oper,
                              Executor* executor) {
  // If here we are operating on top of one or more string functions, i.e. LOWER(str),
  // and before running the dictionary LIKE/ILIKE or REGEXP_LIKE,
  // we need to translate the strings first.

  // This approach is a temporary solution until we can implement the next stage
  // of the string translation project, which will broaden the StringOper class to include
  // operations that operate on strings but do not neccessarily return strings like
  // LIKE/ILIKE/REGEXP_LIKE/CHAR_LENGTH At this point these aforementioned operators,
  // including LIKE/ILIKE, will just become part of a StringOps chain (which will also
  // avoid the overhead of serializing the transformed raw strings from previous string
  // opers to the dictionary to only read back out and perform LIKE/ILIKE.)
  CHECK_GT(string_oper->getArity(), 0UL);
  const auto& string_oper_primary_arg_ti = string_oper->getArg(0)->get_type_info();
  CHECK(string_oper_primary_arg_ti.is_dict_encoded_string());
  CHECK_NE(string_oper_primary_arg_ti.get_comp_param(), TRANSIENT_DICT_ID);
  // Note the actual translation below will be cached by RowSetMemOwner
  translate_dict_strings(string_oper, ExecutorDeviceType::CPU, executor);
}

llvm::Value* CodeGenerator::codegenDictLike(
    const std::shared_ptr<Analyzer::Expr> like_arg,
    const Analyzer::Constant* pattern,
    const bool ilike,
    const bool is_simple,
    const char escape_char,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto cast_oper = std::dynamic_pointer_cast<Analyzer::UOper>(like_arg);
  if (!cast_oper) {
    return nullptr;
  }
  CHECK(cast_oper);
  CHECK_EQ(kCAST, cast_oper->get_optype());
  const auto dict_like_arg = cast_oper->get_own_operand();
  const auto& dict_like_arg_ti = dict_like_arg->get_type_info();
  if (!dict_like_arg_ti.is_string()) {
    throw(std::runtime_error("Cast from " + dict_like_arg_ti.get_type_name() + " to " +
                             cast_oper->get_type_info().get_type_name() +
                             " not supported"));
  }
  CHECK_EQ(kENCODING_DICT, dict_like_arg_ti.get_compression());
  const auto sdp = executor()->getStringDictionaryProxy(
      dict_like_arg_ti.get_comp_param(), executor()->getRowSetMemoryOwner(), true);
  if (sdp->storageEntryCount() > 200000000) {
    return nullptr;
  }
  if (sdp->getDictId() == TRANSIENT_DICT_ID) {
    // If we have a literal dictionary it was a product
    // of string ops applied to none-encoded strings, and
    // will not be populated at codegen-time, so we
    // cannot use the fast path

    // Todo(todd): Once string ops support non-string producting
    // operators (like like/ilike), like/ilike can be chained and
    // we can avoid the string translation
    CHECK(dynamic_cast<const Analyzer::StringOper*>(dict_like_arg.get()));
    return nullptr;
  }
  const auto string_oper = dynamic_cast<const Analyzer::StringOper*>(dict_like_arg.get());
  if (string_oper) {
    pre_translate_string_ops(string_oper, executor());
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
  const auto in_values = std::make_shared<Analyzer::InIntegerSet>(
      dict_like_arg, matching_ids_64, dict_like_arg_ti.get_notnull());
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

llvm::Value* CodeGenerator::codegenDictStrCmp(const std::shared_ptr<Analyzer::Expr> lhs,
                                              const std::shared_ptr<Analyzer::Expr> rhs,
                                              const SQLOps compare_operator,
                                              const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto rhs_cast_oper = std::dynamic_pointer_cast<const Analyzer::UOper>(rhs);
  auto lhs_cast_oper = std::dynamic_pointer_cast<const Analyzer::UOper>(lhs);
  auto rhs_col_var = std::dynamic_pointer_cast<const Analyzer::ColumnVar>(rhs);
  auto lhs_col_var = std::dynamic_pointer_cast<const Analyzer::ColumnVar>(lhs);
  std::shared_ptr<const Analyzer::UOper> cast_oper;
  std::shared_ptr<const Analyzer::ColumnVar> col_var;
  auto compare_opr = compare_operator;
  if (lhs_col_var && rhs_col_var) {
    if (lhs_col_var->get_type_info().get_comp_param() ==
        rhs_col_var->get_type_info().get_comp_param()) {
      if (compare_operator == kEQ || compare_operator == kNE) {
        // TODO (vraj): implement compare between two dictionary encoded columns which
        // share a dictionary
        return nullptr;
      }
    }
    // TODO (vraj): implement compare between two dictionary encoded columns which don't
    // shared dictionary
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

  const auto const_expr =
      dynamic_cast<Analyzer::Constant*>(cast_oper->get_own_operand().get());
  if (!const_expr) {
    // Analyzer casts dictionary encoded columns to none encoded if there is a comparison
    // between two encoded columns. Which we currently do not handle.
    return nullptr;
  }
  const auto& const_val = const_expr->get_constval();

  const auto col_ti = col_var->get_type_info();
  CHECK(col_ti.is_string());
  CHECK_EQ(kENCODING_DICT, col_ti.get_compression());
  const auto sdp = executor()->getStringDictionaryProxy(
      col_ti.get_comp_param(), executor()->getRowSetMemoryOwner(), true);

  if (sdp->storageEntryCount() > 200000000) {
    std::runtime_error("Cardinality for string dictionary is too high");
    return nullptr;
  }

  const auto& pattern_str = *const_val.stringval;
  const auto matching_ids = get_compared_ids(sdp, compare_opr, pattern_str);

  // InIntegerSet requires 64-bit values
  std::vector<int64_t> matching_ids_64(matching_ids.size());
  std::copy(matching_ids.begin(), matching_ids.end(), matching_ids_64.begin());

  const auto in_values = std::make_shared<Analyzer::InIntegerSet>(
      col_var, matching_ids_64, col_ti.get_notnull());
  return codegen(in_values.get(), co);
}

llvm::Value* CodeGenerator::codegen(const Analyzer::RegexpExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (is_unnest(extract_cast_arg(expr->get_arg()))) {
    throw std::runtime_error("REGEXP not supported for unnested expressions");
  }
  char escape_char{'\\'};
  if (expr->get_escape_expr()) {
    auto escape_char_expr =
        dynamic_cast<const Analyzer::Constant*>(expr->get_escape_expr());
    CHECK(escape_char_expr);
    CHECK(escape_char_expr->get_type_info().is_string());
    CHECK_EQ(size_t(1), escape_char_expr->get_constval().stringval->size());
    escape_char = (*escape_char_expr->get_constval().stringval)[0];
  }
  auto pattern = dynamic_cast<const Analyzer::Constant*>(expr->get_pattern_expr());
  CHECK(pattern);
  auto fast_dict_pattern_lv =
      codegenDictRegexp(expr->get_own_arg(), pattern, escape_char, co);
  if (fast_dict_pattern_lv) {
    return fast_dict_pattern_lv;
  }
  const auto& ti = expr->get_arg()->get_type_info();
  CHECK(ti.is_string());
  if (g_enable_watchdog && ti.get_compression() != kENCODING_NONE) {
    throw WatchdogException(
        "Cannot do REGEXP_LIKE on this dictionary encoded column, its cardinality is too "
        "high");
  }
  // Now we know we are working on NONE ENCODED column. So switch back to CPU
  if (co.device_type == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
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
  std::vector<llvm::Value*> regexp_args{
      str_lv[1], str_lv[2], regexp_expr_arg_lvs[1], regexp_expr_arg_lvs[2]};
  std::string fn_name("regexp_like");
  regexp_args.push_back(cgen_state_->llInt(int8_t(escape_char)));
  if (is_nullable) {
    fn_name += "_nullable";
    regexp_args.push_back(cgen_state_->inlineIntNull(expr->get_type_info()));
    return cgen_state_->emitExternalCall(
        fn_name, get_int_type(8, cgen_state_->context_), regexp_args);
  }
  return cgen_state_->emitExternalCall(
      fn_name, get_int_type(1, cgen_state_->context_), regexp_args);
}

llvm::Value* CodeGenerator::codegenDictRegexp(
    const std::shared_ptr<Analyzer::Expr> pattern_arg,
    const Analyzer::Constant* pattern,
    const char escape_char,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
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
  const auto comp_param = dict_regexp_arg_ti.get_comp_param();
  const auto sdp = executor()->getStringDictionaryProxy(
      comp_param, executor()->getRowSetMemoryOwner(), true);
  if (sdp->storageEntryCount() > 15000000) {
    return nullptr;
  }
  if (sdp->getDictId() == TRANSIENT_DICT_ID) {
    // If we have a literal dictionary it was a product
    // of string ops applied to none-encoded strings, and
    // will not be populated at codegen-time, so we
    // cannot use the fast path

    // Todo(todd): Once string ops support non-string producting
    // operators (like regexp_like), these operators can be chained
    // and we can avoid the string translation
    return nullptr;
  }
  const auto string_oper =
      dynamic_cast<const Analyzer::StringOper*>(dict_regexp_arg.get());
  if (string_oper) {
    pre_translate_string_ops(string_oper, executor());
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
  const auto in_values = std::make_shared<Analyzer::InIntegerSet>(
      dict_regexp_arg, matching_ids_64, dict_regexp_arg_ti.get_notnull());
  return codegen(in_values.get(), co);
}
