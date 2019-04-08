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

#include <typeinfo>

#include "../Parser/ParserNode.h"

namespace {

llvm::CmpInst::Predicate llvm_icmp_pred(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return llvm::ICmpInst::ICMP_EQ;
    case kNE:
      return llvm::ICmpInst::ICMP_NE;
    case kLT:
      return llvm::ICmpInst::ICMP_SLT;
    case kGT:
      return llvm::ICmpInst::ICMP_SGT;
    case kLE:
      return llvm::ICmpInst::ICMP_SLE;
    case kGE:
      return llvm::ICmpInst::ICMP_SGE;
    default:
      abort();
  }
}

std::string icmp_name(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return "eq";
    case kNE:
      return "ne";
    case kLT:
      return "lt";
    case kGT:
      return "gt";
    case kLE:
      return "le";
    case kGE:
      return "ge";
    default:
      abort();
  }
}

std::string icmp_arr_name(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return "eq";
    case kNE:
      return "ne";
    case kLT:
      return "gt";
    case kGT:
      return "lt";
    case kLE:
      return "ge";
    case kGE:
      return "le";
    default:
      abort();
  }
}

llvm::CmpInst::Predicate llvm_fcmp_pred(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return llvm::CmpInst::FCMP_OEQ;
    case kNE:
      return llvm::CmpInst::FCMP_ONE;
    case kLT:
      return llvm::CmpInst::FCMP_OLT;
    case kGT:
      return llvm::CmpInst::FCMP_OGT;
    case kLE:
      return llvm::CmpInst::FCMP_OLE;
    case kGE:
      return llvm::CmpInst::FCMP_OGE;
    default:
      abort();
  }
}

}  // namespace

namespace {

std::string string_cmp_func(const SQLOps optype) {
  switch (optype) {
    case kLT:
      return "string_lt";
    case kLE:
      return "string_le";
    case kGT:
      return "string_gt";
    case kGE:
      return "string_ge";
    case kEQ:
      return "string_eq";
    case kNE:
      return "string_ne";
    default:
      abort();
  }
}

std::shared_ptr<Analyzer::BinOper> lower_bw_eq(const Analyzer::BinOper* bw_eq) {
  const auto eq_oper =
      std::make_shared<Analyzer::BinOper>(bw_eq->get_type_info(),
                                          bw_eq->get_contains_agg(),
                                          kEQ,
                                          bw_eq->get_qualifier(),
                                          bw_eq->get_own_left_operand(),
                                          bw_eq->get_own_right_operand());
  const auto lhs_is_null =
      std::make_shared<Analyzer::UOper>(kBOOLEAN, kISNULL, bw_eq->get_own_left_operand());
  const auto rhs_is_null = std::make_shared<Analyzer::UOper>(
      kBOOLEAN, kISNULL, bw_eq->get_own_right_operand());
  const auto both_are_null =
      Parser::OperExpr::normalize(kAND, kONE, lhs_is_null, rhs_is_null);
  const auto bw_eq_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(
      Parser::OperExpr::normalize(kOR, kONE, eq_oper, both_are_null));
  CHECK(bw_eq_oper);
  return bw_eq_oper;
}

std::shared_ptr<Analyzer::BinOper> make_eq(const std::shared_ptr<Analyzer::Expr>& lhs,
                                           const std::shared_ptr<Analyzer::Expr>& rhs,
                                           const SQLOps optype) {
  CHECK(IS_EQUIVALENCE(optype));
  // Sides of a tuple equality are stripped of cast operators to simplify the logic
  // in the hash table construction algorithm. Add them back here.
  auto eq_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(
      Parser::OperExpr::normalize(optype, kONE, lhs, rhs));
  CHECK(eq_oper);
  return optype == kBW_EQ ? lower_bw_eq(eq_oper.get()) : eq_oper;
}

// Convert a column tuple equality expression back to a conjunction of comparisons
// so that it can be handled by the regular code generation methods.
std::shared_ptr<Analyzer::BinOper> lower_multicol_compare(
    const Analyzer::BinOper* multicol_compare) {
  const auto left_tuple_expr = dynamic_cast<const Analyzer::ExpressionTuple*>(
      multicol_compare->get_left_operand());
  const auto right_tuple_expr = dynamic_cast<const Analyzer::ExpressionTuple*>(
      multicol_compare->get_right_operand());
  CHECK(left_tuple_expr && right_tuple_expr);
  const auto& left_tuple = left_tuple_expr->getTuple();
  const auto& right_tuple = right_tuple_expr->getTuple();
  CHECK_EQ(left_tuple.size(), right_tuple.size());
  CHECK_GT(left_tuple.size(), size_t(1));
  auto acc =
      make_eq(left_tuple.front(), right_tuple.front(), multicol_compare->get_optype());
  for (size_t i = 1; i < left_tuple.size(); ++i) {
    auto crt = make_eq(left_tuple[i], right_tuple[i], multicol_compare->get_optype());
    const bool not_null =
        acc->get_type_info().get_notnull() && crt->get_type_info().get_notnull();
    acc = makeExpr<Analyzer::BinOper>(
        SQLTypeInfo(kBOOLEAN, not_null), false, kAND, kONE, acc, crt);
  }
  return acc;
}

}  // namespace

llvm::Value* Executor::codegenCmp(const Analyzer::BinOper* bin_oper,
                                  const CompilationOptions& co) {
  const auto qualifier = bin_oper->get_qualifier();
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  if (dynamic_cast<const Analyzer::ExpressionTuple*>(lhs)) {
    CHECK(dynamic_cast<const Analyzer::ExpressionTuple*>(rhs));
    const auto lowered = lower_multicol_compare(bin_oper);
    const auto lowered_lvs = codegen(lowered.get(), true, co);
    CHECK_EQ(size_t(1), lowered_lvs.size());
    return lowered_lvs.front();
  }
  const auto optype = bin_oper->get_optype();
  if (optype == kBW_EQ) {
    const auto bw_eq_oper = lower_bw_eq(bin_oper);
    return codegenLogical(bw_eq_oper.get(), co);
  }
  if (optype == kOVERLAPS) {
    return codegenOverlaps(optype,
                           qualifier,
                           bin_oper->get_own_left_operand(),
                           bin_oper->get_own_right_operand(),
                           co);
  }
  if (is_unnest(lhs) || is_unnest(rhs)) {
    throw std::runtime_error("Unnest not supported in comparisons");
  }
  const auto& lhs_ti = lhs->get_type_info();
  const auto& rhs_ti = rhs->get_type_info();

  if (lhs_ti.is_string() && rhs_ti.is_string() &&
      !(IS_EQUIVALENCE(optype) || optype == kNE)) {
    auto cmp_str = codegenStrCmp(optype,
                                 qualifier,
                                 bin_oper->get_own_left_operand(),
                                 bin_oper->get_own_right_operand(),
                                 co);
    if (cmp_str) {
      return cmp_str;
    }
  }

  if (lhs_ti.is_decimal()) {
    auto cmp_decimal_const =
        codegenCmpDecimalConst(optype, qualifier, lhs, lhs_ti, rhs, co);
    if (cmp_decimal_const) {
      return cmp_decimal_const;
    }
  }

  auto lhs_lvs = codegen(lhs, true, co);
  return codegenCmp(optype, qualifier, lhs_lvs, lhs_ti, rhs, co);
}

llvm::Value* Executor::codegenOverlaps(const SQLOps optype,
                                       const SQLQualifier qualifier,
                                       const std::shared_ptr<Analyzer::Expr> lhs,
                                       const std::shared_ptr<Analyzer::Expr> rhs,
                                       const CompilationOptions& co) {
  // TODO(adb): we should never get here, but going to leave this in place for now since
  // it will likely be useful in factoring the bounds check out of ST_Contains
  const auto lhs_ti = lhs->get_type_info();
  CHECK(lhs_ti.is_geometry());

  if (lhs_ti.is_geometry()) {
    // only point in linestring/poly/mpoly is currently supported
    CHECK(lhs_ti.get_type() == kPOINT);
    const auto lhs_col = dynamic_cast<Analyzer::ColumnVar*>(lhs.get());
    CHECK(lhs_col);

    // Get the actual point data column descriptor
    const auto coords_cd = catalog_->getMetadataForColumn(lhs_col->get_table_id(),
                                                          lhs_col->get_column_id() + 1);
    CHECK(coords_cd);

    std::vector<std::shared_ptr<Analyzer::Expr>> geoargs;
    geoargs.push_back(makeExpr<Analyzer::ColumnVar>(coords_cd->columnType,
                                                    coords_cd->tableId,
                                                    coords_cd->columnId,
                                                    lhs_col->get_rte_idx()));

    Datum input_compression;
    input_compression.intval =
        (lhs_ti.get_compression() == kENCODING_GEOINT && lhs_ti.get_comp_param() == 32)
            ? 1
            : 0;
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_compression));
    Datum input_srid;
    input_srid.intval = lhs_ti.get_input_srid();
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, input_srid));
    Datum output_srid;
    output_srid.intval = lhs_ti.get_output_srid();
    geoargs.push_back(makeExpr<Analyzer::Constant>(kINT, false, output_srid));

    const auto x_ptr_oper = makeExpr<Analyzer::FunctionOper>(
        SQLTypeInfo(kDOUBLE, true), "ST_X_Point", geoargs);
    const auto y_ptr_oper = makeExpr<Analyzer::FunctionOper>(
        SQLTypeInfo(kDOUBLE, true), "ST_Y_Point", geoargs);

    const auto rhs_ti = rhs->get_type_info();
    CHECK(IS_GEO_POLY(rhs_ti.get_type()));
    const auto rhs_col = dynamic_cast<Analyzer::ColumnVar*>(rhs.get());
    CHECK(rhs_col);

    const auto poly_bounds_cd = catalog_->getMetadataForColumn(
        rhs_col->get_table_id(),
        rhs_col->get_column_id() + rhs_ti.get_physical_coord_cols() + 1);
    CHECK(poly_bounds_cd);

    auto bbox_col_var = makeExpr<Analyzer::ColumnVar>(poly_bounds_cd->columnType,
                                                      poly_bounds_cd->tableId,
                                                      poly_bounds_cd->columnId,
                                                      rhs_col->get_rte_idx());

    const auto bbox_contains_func_oper =
        makeExpr<Analyzer::FunctionOper>(SQLTypeInfo(kBOOLEAN, false),
                                         "Point_Overlaps_Box",
                                         std::vector<std::shared_ptr<Analyzer::Expr>>{
                                             bbox_col_var, x_ptr_oper, y_ptr_oper});

    return codegenFunctionOper(bbox_contains_func_oper.get(), co);
  }

  CHECK(false) << "Unsupported type for overlaps operator: " << lhs_ti.get_type_name();
  return nullptr;
}

llvm::Value* Executor::codegenStrCmp(const SQLOps optype,
                                     const SQLQualifier qualifier,
                                     const std::shared_ptr<Analyzer::Expr> lhs,
                                     const std::shared_ptr<Analyzer::Expr> rhs,
                                     const CompilationOptions& co) {
  const auto lhs_ti = lhs->get_type_info();
  const auto rhs_ti = rhs->get_type_info();

  CHECK(lhs_ti.is_string());
  CHECK(rhs_ti.is_string());

  const auto null_check_suffix = get_null_check_suffix(lhs_ti, rhs_ti);
  if (lhs_ti.get_compression() == kENCODING_DICT &&
      rhs_ti.get_compression() == kENCODING_DICT) {
    if (lhs_ti.get_comp_param() == rhs_ti.get_comp_param()) {
      // Both operands share a dictionary

      // check if query is trying to compare a columnt against literal

      auto ir = codegenDictStrCmp(lhs, rhs, optype, co);
      if (ir) {
        return ir;
      }
    } else {
      // Both operands don't share a dictionary
      return nullptr;
    }
  }
  return nullptr;
}
llvm::Value* Executor::codegenCmpDecimalConst(const SQLOps optype,
                                              const SQLQualifier qualifier,
                                              const Analyzer::Expr* lhs,
                                              const SQLTypeInfo& lhs_ti,
                                              const Analyzer::Expr* rhs,
                                              const CompilationOptions& co) {
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(lhs);
  if (!u_oper || u_oper->get_optype() != kCAST) {
    return nullptr;
  }
  auto rhs_constant = dynamic_cast<const Analyzer::Constant*>(rhs);
  if (!rhs_constant) {
    return nullptr;
  }
  const auto operand = u_oper->get_operand();
  const auto& operand_ti = operand->get_type_info();
  if (operand_ti.is_decimal() && operand_ti.get_scale() < lhs_ti.get_scale()) {
    // lhs decimal type has smaller scale
  } else if (operand_ti.is_integer() && 0 < lhs_ti.get_scale()) {
    // lhs is integer, no need to scale it all the way up to the cmp expr scale
  } else {
    return nullptr;
  }

  auto scale_diff = lhs_ti.get_scale() - operand_ti.get_scale() - 1;
  int64_t bigintval = rhs_constant->get_constval().bigintval;
  bool negative = false;
  if (bigintval < 0) {
    negative = true;
    bigintval = -bigintval;
  }
  int64_t truncated_decimal = bigintval / exp_to_scale(scale_diff);
  int64_t decimal_tail = bigintval % exp_to_scale(scale_diff);
  if (truncated_decimal % 10 == 0 && decimal_tail > 0) {
    truncated_decimal += 1;
  }
  SQLTypeInfo new_ti = SQLTypeInfo(
      kDECIMAL, 19, lhs_ti.get_scale() - scale_diff, operand_ti.get_notnull());
  if (negative) {
    truncated_decimal = -truncated_decimal;
  }
  Datum d;
  d.bigintval = truncated_decimal;
  const auto new_rhs_lit =
      makeExpr<Analyzer::Constant>(new_ti, rhs_constant->get_is_null(), d);
  const auto operand_lv = codegen(operand, true, co).front();
  const auto lhs_lv = codegenCast(operand_lv, operand_ti, new_ti, false, co);
  return codegenCmp(optype, qualifier, {lhs_lv}, new_ti, new_rhs_lit.get(), co);
}

llvm::Value* Executor::codegenCmp(const SQLOps optype,
                                  const SQLQualifier qualifier,
                                  std::vector<llvm::Value*> lhs_lvs,
                                  const SQLTypeInfo& lhs_ti,
                                  const Analyzer::Expr* rhs,
                                  const CompilationOptions& co) {
  CHECK(IS_COMPARISON(optype));
  const auto& rhs_ti = rhs->get_type_info();
  if (rhs_ti.is_array()) {
    return codegenQualifierCmp(optype, qualifier, lhs_lvs, rhs, co);
  }
  auto rhs_lvs = codegen(rhs, true, co);
  CHECK_EQ(kONE, qualifier);
  if (optype == kOVERLAPS) {
    CHECK(lhs_ti.is_geometry());
    CHECK(rhs_ti.is_array() ||
          rhs_ti.is_geometry());  // allow geo col or bounds col to pass
  } else {
    CHECK((lhs_ti.get_type() == rhs_ti.get_type()) ||
          (lhs_ti.is_string() && rhs_ti.is_string()));
  }
  const auto null_check_suffix = get_null_check_suffix(lhs_ti, rhs_ti);
  if (lhs_ti.is_integer() || lhs_ti.is_decimal() || lhs_ti.is_time() ||
      lhs_ti.is_boolean() || lhs_ti.is_string() || lhs_ti.is_timeinterval()) {
    if (lhs_ti.is_string()) {
      CHECK(rhs_ti.is_string());
      CHECK_EQ(lhs_ti.get_compression(), rhs_ti.get_compression());
      if (lhs_ti.get_compression() == kENCODING_NONE) {
        // unpack pointer + length if necessary
        if (lhs_lvs.size() != 3) {
          CHECK_EQ(size_t(1), lhs_lvs.size());
          lhs_lvs.push_back(cgen_state_->emitCall("extract_str_ptr", {lhs_lvs.front()}));
          lhs_lvs.push_back(cgen_state_->emitCall("extract_str_len", {lhs_lvs.front()}));
        }
        if (rhs_lvs.size() != 3) {
          CHECK_EQ(size_t(1), rhs_lvs.size());
          rhs_lvs.push_back(cgen_state_->emitCall("extract_str_ptr", {rhs_lvs.front()}));
          rhs_lvs.push_back(cgen_state_->emitCall("extract_str_len", {rhs_lvs.front()}));
        }
        std::vector<llvm::Value*> str_cmp_args{
            lhs_lvs[1], lhs_lvs[2], rhs_lvs[1], rhs_lvs[2]};
        if (!null_check_suffix.empty()) {
          str_cmp_args.push_back(inlineIntNull(SQLTypeInfo(kBOOLEAN, false)));
        }
        return cgen_state_->emitCall(
            string_cmp_func(optype) + (null_check_suffix.empty() ? "" : "_nullable"),
            str_cmp_args);
      } else {
        CHECK(optype == kEQ || optype == kNE);
      }
    }
    return null_check_suffix.empty()
               ? cgen_state_->ir_builder_.CreateICmp(
                     llvm_icmp_pred(optype), lhs_lvs.front(), rhs_lvs.front())
               : cgen_state_->emitCall(icmp_name(optype) + "_" +
                                           numeric_type_name(lhs_ti) + null_check_suffix,
                                       {lhs_lvs.front(),
                                        rhs_lvs.front(),
                                        ll_int(inline_int_null_val(lhs_ti)),
                                        inlineIntNull(SQLTypeInfo(kBOOLEAN, false))});
  }
  if (lhs_ti.get_type() == kFLOAT || lhs_ti.get_type() == kDOUBLE) {
    return null_check_suffix.empty()
               ? cgen_state_->ir_builder_.CreateFCmp(
                     llvm_fcmp_pred(optype), lhs_lvs.front(), rhs_lvs.front())
               : cgen_state_->emitCall(icmp_name(optype) + "_" +
                                           numeric_type_name(lhs_ti) + null_check_suffix,
                                       {lhs_lvs.front(),
                                        rhs_lvs.front(),
                                        lhs_ti.get_type() == kFLOAT ? ll_fp(NULL_FLOAT)
                                                                    : ll_fp(NULL_DOUBLE),
                                        inlineIntNull(SQLTypeInfo(kBOOLEAN, false))});
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* Executor::codegenQualifierCmp(const SQLOps optype,
                                           const SQLQualifier qualifier,
                                           std::vector<llvm::Value*> lhs_lvs,
                                           const Analyzer::Expr* rhs,
                                           const CompilationOptions& co) {
  const auto& rhs_ti = rhs->get_type_info();
  const Analyzer::Expr* arr_expr{rhs};
  if (dynamic_cast<const Analyzer::UOper*>(rhs)) {
    const auto cast_arr = static_cast<const Analyzer::UOper*>(rhs);
    CHECK_EQ(kCAST, cast_arr->get_optype());
    arr_expr = cast_arr->get_operand();
  }
  const auto& arr_ti = arr_expr->get_type_info();
  const auto& elem_ti = arr_ti.get_elem_type();
  auto rhs_lvs = codegen(arr_expr, true, co);
  CHECK_NE(kONE, qualifier);
  std::string fname{std::string("array_") + (qualifier == kANY ? "any" : "all") + "_" +
                    icmp_arr_name(optype)};
  const auto& target_ti = rhs_ti.get_elem_type();
  const bool is_real_string{target_ti.is_string() &&
                            target_ti.get_compression() != kENCODING_DICT};
  if (is_real_string) {
    if (g_cluster) {
      throw std::runtime_error(
          "Comparison between a dictionary-encoded and a none-encoded string not "
          "supported for distributed queries");
    }
    if (g_enable_watchdog) {
      throw WatchdogException(
          "Comparison between a dictionary-encoded and a none-encoded string would be "
          "slow");
    }
    if (co.device_type_ == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    CHECK_EQ(kENCODING_NONE, target_ti.get_compression());
    fname += "_str";
  }
  if (elem_ti.is_integer() || elem_ti.is_boolean() || elem_ti.is_string()) {
    fname += ("_" + numeric_type_name(elem_ti));
  } else {
    CHECK(elem_ti.is_fp());
    fname += elem_ti.get_type() == kDOUBLE ? "_double" : "_float";
  }
  if (is_real_string) {
    CHECK_EQ(size_t(3), lhs_lvs.size());
    return cgen_state_->emitExternalCall(
        fname,
        get_int_type(1, cgen_state_->context_),
        {rhs_lvs.front(),
         posArg(arr_expr),
         lhs_lvs[1],
         lhs_lvs[2],
         ll_int(int64_t(getStringDictionaryProxy(
             elem_ti.get_comp_param(), row_set_mem_owner_, true))),
         inlineIntNull(elem_ti)});
  }
  if (target_ti.is_integer() || target_ti.is_boolean() || target_ti.is_string()) {
    fname += ("_" + numeric_type_name(target_ti));
  } else {
    CHECK(target_ti.is_fp());
    fname += target_ti.get_type() == kDOUBLE ? "_double" : "_float";
  }
  return cgen_state_->emitExternalCall(
      fname,
      get_int_type(1, cgen_state_->context_),
      {rhs_lvs.front(),
       posArg(arr_expr),
       lhs_lvs.front(),
       elem_ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(elem_ti))
                       : static_cast<llvm::Value*>(inlineIntNull(elem_ti))});
}
