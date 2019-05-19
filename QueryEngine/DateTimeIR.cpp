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

#include "DateTimeUtils.h"
#include "Execute.h"

using namespace DateTimeUtils;

llvm::Value* CodeGenerator::codegen(const Analyzer::ExtractExpr* extract_expr,
                                    const CompilationOptions& co) {
  auto from_expr = codegen(extract_expr->get_from_expr(), true, co).front();
  const int32_t extract_field{extract_expr->get_field()};
  const auto& extract_expr_ti = extract_expr->get_from_expr()->get_type_info();
  if (extract_field == kEPOCH) {
    CHECK(extract_expr_ti.get_type() == kTIMESTAMP ||
          extract_expr_ti.get_type() == kDATE);
    if (from_expr->getType()->isIntegerTy(32)) {
      from_expr =
          cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
                                              from_expr,
                                              get_int_type(64, cgen_state_->context_));
    }
    return from_expr;
  }
  CHECK(from_expr->getType()->isIntegerTy(64));
  if (extract_expr_ti.is_high_precision_timestamp()) {
    from_expr = codegenExtractHighPrecisionTimestamps(
        from_expr, extract_expr_ti, extract_expr->get_field());
  }
  if (!extract_expr_ti.is_high_precision_timestamp() &&
      is_subsecond_extract_field(extract_expr->get_field())) {
    from_expr =
        extract_expr_ti.get_notnull()
            ? cgen_state_->ir_builder_.CreateMul(
                  from_expr,
                  cgen_state_->llInt(
                      get_extract_timestamp_precision_scale(extract_expr->get_field())))
            : cgen_state_->emitCall(
                  "mul_int64_t_nullable_lhs",
                  {from_expr,
                   cgen_state_->llInt(
                       get_extract_timestamp_precision_scale(extract_expr->get_field())),
                   cgen_state_->inlineIntNull(extract_expr_ti)});
  }
  std::vector<llvm::Value*> extract_args{
      cgen_state_->llInt(static_cast<int32_t>(extract_expr->get_field())), from_expr};
  std::string extract_fname{"ExtractFromTime"};
  if (!extract_expr_ti.get_notnull()) {
    extract_args.push_back(cgen_state_->inlineIntNull(extract_expr_ti));
    extract_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      extract_fname, get_int_type(64, cgen_state_->context_), extract_args);
}

llvm::Value* CodeGenerator::codegen(const Analyzer::DateaddExpr* dateadd_expr,
                                    const CompilationOptions& co) {
  const auto& dateadd_expr_ti = dateadd_expr->get_type_info();
  CHECK(dateadd_expr_ti.get_type() == kTIMESTAMP || dateadd_expr_ti.get_type() == kDATE);
  auto datetime = codegen(dateadd_expr->get_datetime_expr(), true, co).front();
  CHECK(datetime->getType()->isIntegerTy(64));
  auto number = codegen(dateadd_expr->get_number_expr(), true, co).front();

  const auto& datetime_ti = dateadd_expr->get_datetime_expr()->get_type_info();
  std::vector<llvm::Value*> dateadd_args{
      cgen_state_->llInt(static_cast<int32_t>(dateadd_expr->get_field())),
      number,
      datetime};
  std::string dateadd_fname{"DateAdd"};
  if (dateadd_expr_ti.is_high_precision_timestamp()) {
    dateadd_fname += "HighPrecision";
    dateadd_args.push_back(cgen_state_->llInt(static_cast<int64_t>(
        get_timestamp_precision_scale(dateadd_expr_ti.get_dimension()))));
  }
  if (!datetime_ti.get_notnull()) {
    dateadd_args.push_back(cgen_state_->inlineIntNull(datetime_ti));
    dateadd_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      dateadd_fname, get_int_type(64, cgen_state_->context_), dateadd_args);
}

llvm::Value* CodeGenerator::codegen(const Analyzer::DatediffExpr* datediff_expr,
                                    const CompilationOptions& co) {
  auto start = codegen(datediff_expr->get_start_expr(), true, co).front();
  CHECK(start->getType()->isIntegerTy(64));
  auto end = codegen(datediff_expr->get_end_expr(), true, co).front();
  CHECK(end->getType()->isIntegerTy(32) || end->getType()->isIntegerTy(64));
  const auto& start_ti = datediff_expr->get_start_expr()->get_type_info();
  const auto& end_ti = datediff_expr->get_end_expr()->get_type_info();
  std::vector<llvm::Value*> datediff_args{
      cgen_state_->llInt(static_cast<int32_t>(datediff_expr->get_field())), start, end};
  std::string datediff_fname{"DateDiff"};
  if (start_ti.is_high_precision_timestamp() || end_ti.is_high_precision_timestamp()) {
    const auto adj_dimen = end_ti.get_dimension() - start_ti.get_dimension();
    datediff_fname += "HighPrecision";
    datediff_args.push_back(cgen_state_->llInt(static_cast<int32_t>(adj_dimen)));
    datediff_args.push_back(cgen_state_->llInt(
        static_cast<int64_t>(get_timestamp_precision_scale(abs(adj_dimen)))));
    datediff_args.push_back(
        cgen_state_->llInt(static_cast<int64_t>(get_timestamp_precision_scale(
            adj_dimen < 0 ? end_ti.get_dimension() : start_ti.get_dimension()))));
    datediff_args.push_back(
        cgen_state_->llInt(static_cast<int64_t>(get_timestamp_precision_scale(
            adj_dimen > 0 ? end_ti.get_dimension() : start_ti.get_dimension()))));
  }
  const auto& ret_ti = datediff_expr->get_type_info();
  if (!start_ti.get_notnull() || !end_ti.get_notnull()) {
    datediff_args.push_back(cgen_state_->inlineIntNull(ret_ti));
    datediff_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      datediff_fname, get_int_type(64, cgen_state_->context_), datediff_args);
}

llvm::Value* CodeGenerator::codegen(const Analyzer::DatetruncExpr* datetrunc_expr,
                                    const CompilationOptions& co) {
  auto from_expr = codegen(datetrunc_expr->get_from_expr(), true, co).front();
  const auto& datetrunc_expr_ti = datetrunc_expr->get_from_expr()->get_type_info();
  CHECK(from_expr->getType()->isIntegerTy(64));
  if (datetrunc_expr_ti.is_high_precision_timestamp()) {
    return codegenDateTruncHighPrecisionTimestamps(
        from_expr, datetrunc_expr_ti, datetrunc_expr->get_field());
  }
  std::vector<llvm::Value*> datetrunc_args{
      cgen_state_->llInt(static_cast<int32_t>(datetrunc_expr->get_field())), from_expr};
  std::string datetrunc_fname{"DateTruncate"};
  if (!datetrunc_expr_ti.get_notnull()) {
    datetrunc_args.push_back(cgen_state_->inlineIntNull(datetrunc_expr_ti));
    datetrunc_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      datetrunc_fname, get_int_type(64, cgen_state_->context_), datetrunc_args);
}

llvm::Value* CodeGenerator::codegenExtractHighPrecisionTimestamps(
    llvm::Value* ts_lv,
    const SQLTypeInfo& ti,
    const ExtractField& field) {
  CHECK(ti.is_high_precision_timestamp());
  CHECK(ts_lv->getType()->isIntegerTy(64));
  if (is_subsecond_extract_field(field)) {
    const auto result =
        get_extract_high_precision_adjusted_scale(field, ti.get_dimension());
    if (result.first == kMULTIPLY) {
      return ti.get_notnull()
                 ? cgen_state_->ir_builder_.CreateMul(
                       ts_lv, cgen_state_->llInt(static_cast<int64_t>(result.second)))
                 : cgen_state_->emitCall(
                       "mul_int64_t_nullable_lhs",
                       {ts_lv,
                        cgen_state_->llInt(static_cast<int64_t>(result.second)),
                        cgen_state_->inlineIntNull(ti)});
    } else if (result.first == kDIVIDE) {
      return ti.get_notnull()
                 ? cgen_state_->ir_builder_.CreateSDiv(
                       ts_lv, cgen_state_->llInt(static_cast<int64_t>(result.second)))
                 : cgen_state_->emitCall(
                       "div_int64_t_nullable_lhs",
                       {ts_lv,
                        cgen_state_->llInt(static_cast<int64_t>(result.second)),
                        cgen_state_->inlineIntNull(ti)});
    } else {
      return ts_lv;
    }
  }
  return ti.get_notnull()
             ? cgen_state_->ir_builder_.CreateSDiv(
                   ts_lv,
                   cgen_state_->llInt(static_cast<int64_t>(
                       get_timestamp_precision_scale(ti.get_dimension()))))
             : cgen_state_->emitCall(
                   "div_int64_t_nullable_lhs",
                   {ts_lv,
                    cgen_state_->llInt(get_timestamp_precision_scale(ti.get_dimension())),
                    cgen_state_->inlineIntNull(ti)});
}

llvm::Value* CodeGenerator::codegenDateTruncHighPrecisionTimestamps(
    llvm::Value* ts_lv,
    const SQLTypeInfo& ti,
    const DatetruncField& field) {
  CHECK(ti.is_high_precision_timestamp());
  CHECK(ts_lv->getType()->isIntegerTy(64));
  if (is_subsecond_datetrunc_field(field)) {
    const auto result = get_datetrunc_high_precision_scale(field, ti.get_dimension());
    if (result != -1) {
      ts_lv =
          ti.get_notnull()
              ? cgen_state_->ir_builder_.CreateSDiv(
                    ts_lv, cgen_state_->llInt(static_cast<int64_t>(result)))
              : cgen_state_->emitCall("div_int64_t_nullable_lhs",
                                      {ts_lv,
                                       cgen_state_->llInt(static_cast<int64_t>(result)),
                                       cgen_state_->inlineIntNull(ti)});
      return ti.get_notnull()
                 ? cgen_state_->ir_builder_.CreateMul(
                       ts_lv, cgen_state_->llInt(static_cast<int64_t>(result)))
                 : cgen_state_->emitCall(
                       "mul_int64_t_nullable_lhs",
                       {ts_lv,
                        cgen_state_->llInt(static_cast<int64_t>(result)),
                        cgen_state_->inlineIntNull(ti)});
    } else {
      return ts_lv;
    }
  }
  std::string datetrunc_fname = "DateTruncate";
  const int64_t scale = get_timestamp_precision_scale(ti.get_dimension());
  ts_lv = ti.get_notnull()
              ? cgen_state_->ir_builder_.CreateSDiv(
                    ts_lv, cgen_state_->llInt(static_cast<int64_t>(scale)))
              : cgen_state_->emitCall("div_int64_t_nullable_lhs",
                                      {ts_lv,
                                       cgen_state_->llInt(static_cast<int64_t>(scale)),
                                       cgen_state_->inlineIntNull(ti)});
  std::vector<llvm::Value*> datetrunc_args{
      cgen_state_->llInt(static_cast<int32_t>(field)), ts_lv};
  if (!ti.get_notnull()) {
    datetrunc_fname += "Nullable";
    datetrunc_args.push_back(cgen_state_->inlineIntNull(ti));
  }
  ts_lv = cgen_state_->emitExternalCall(
      datetrunc_fname, get_int_type(64, cgen_state_->context_), datetrunc_args);
  return ti.get_notnull()
             ? cgen_state_->ir_builder_.CreateMul(
                   ts_lv, cgen_state_->llInt(static_cast<int64_t>(scale)))
             : cgen_state_->emitCall("mul_int64_t_nullable_lhs",
                                     {ts_lv,
                                      cgen_state_->llInt(static_cast<int64_t>(scale)),
                                      cgen_state_->inlineIntNull(ti)});
}
