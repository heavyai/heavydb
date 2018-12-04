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

llvm::Value* Executor::codegen(const Analyzer::ExtractExpr* extract_expr,
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
  CHECK(from_expr->getType()->isIntegerTy(32) || from_expr->getType()->isIntegerTy(64));
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  if (sizeof(time_t) == 4 && from_expr->getType()->isIntegerTy(64)) {
    from_expr =
        cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::Trunc,
                                            from_expr,
                                            get_int_type(32, cgen_state_->context_));
  }
  std::vector<llvm::Value*> extract_args{
      ll_int(static_cast<int32_t>(extract_expr->get_field())), from_expr};
  std::string extract_fname{"ExtractFromTime"};
  if (extract_expr_ti.get_dimension() > 0) {
    extract_fname += "HighPrecision";
    extract_args.push_back(ll_int(static_cast<int64_t>(
        get_timestamp_precision_scale(extract_expr_ti.get_dimension()))));
  }
  if (!extract_expr_ti.get_notnull()) {
    extract_args.push_back(inlineIntNull(extract_expr_ti));
    extract_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      extract_fname, get_int_type(64, cgen_state_->context_), extract_args);
}

llvm::Value* Executor::codegen(const Analyzer::DateaddExpr* dateadd_expr,
                               const CompilationOptions& co) {
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  const auto& dateadd_expr_ti = dateadd_expr->get_type_info();
  CHECK(dateadd_expr_ti.get_type() == kTIMESTAMP || dateadd_expr_ti.get_type() == kDATE);
  auto datetime = codegen(dateadd_expr->get_datetime_expr(), true, co).front();
  CHECK(datetime->getType()->isIntegerTy(32) || datetime->getType()->isIntegerTy(64));
  if (sizeof(time_t) == 4 && datetime->getType()->isIntegerTy(64)) {
    datetime =
        cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::Trunc,
                                            datetime,
                                            get_int_type(32, cgen_state_->context_));
  }
  auto number = codegen(dateadd_expr->get_number_expr(), true, co).front();

  const auto& datetime_ti = dateadd_expr->get_datetime_expr()->get_type_info();
  std::vector<llvm::Value*> dateadd_args{
      ll_int(static_cast<int32_t>(dateadd_expr->get_field())), number, datetime};
  std::string dateadd_fname{"DateAdd"};
  if (dateadd_expr_ti.is_high_precision_timestamp()) {
    dateadd_fname += "HighPrecision";
    dateadd_args.push_back(ll_int(static_cast<int64_t>(
        get_timestamp_precision_scale(dateadd_expr_ti.get_dimension()))));
  }
  if (!datetime_ti.get_notnull()) {
    dateadd_args.push_back(inlineIntNull(datetime_ti));
    dateadd_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      dateadd_fname, get_int_type(64, cgen_state_->context_), dateadd_args);
}

llvm::Value* Executor::codegen(const Analyzer::DatediffExpr* datediff_expr,
                               const CompilationOptions& co) {
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  auto start = codegen(datediff_expr->get_start_expr(), true, co).front();
  CHECK(start->getType()->isIntegerTy(32) || start->getType()->isIntegerTy(64));
  if (sizeof(time_t) == 4 && start->getType()->isIntegerTy(64)) {
    start = cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::Trunc,
                                                start,
                                                get_int_type(32, cgen_state_->context_));
  }
  auto end = codegen(datediff_expr->get_end_expr(), true, co).front();
  CHECK(end->getType()->isIntegerTy(32) || end->getType()->isIntegerTy(64));
  if (sizeof(time_t) == 4 && end->getType()->isIntegerTy(64)) {
    end = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::Trunc, end, get_int_type(32, cgen_state_->context_));
  }
  const auto& start_ti = datediff_expr->get_start_expr()->get_type_info();
  const auto& end_ti = datediff_expr->get_end_expr()->get_type_info();
  std::vector<llvm::Value*> datediff_args{
      ll_int(static_cast<int32_t>(datediff_expr->get_field())), start, end};
  std::string datediff_fname{"DateDiff"};
  if (start_ti.is_high_precision_timestamp() || end_ti.is_high_precision_timestamp()) {
    const auto adj_dimen = end_ti.get_dimension() - start_ti.get_dimension();
    datediff_fname += "HighPrecision";
    datediff_args.push_back(ll_int(static_cast<int32_t>(adj_dimen)));
    datediff_args.push_back(
        ll_int(static_cast<int64_t>(get_timestamp_precision_scale(abs(adj_dimen)))));
    datediff_args.push_back(ll_int(static_cast<int64_t>(get_timestamp_precision_scale(
        adj_dimen < 0 ? end_ti.get_dimension() : start_ti.get_dimension()))));
    datediff_args.push_back(ll_int(static_cast<int64_t>(get_timestamp_precision_scale(
        adj_dimen > 0 ? end_ti.get_dimension() : start_ti.get_dimension()))));
  }
  const auto& ret_ti = datediff_expr->get_type_info();
  if (!start_ti.get_notnull() || !end_ti.get_notnull()) {
    datediff_args.push_back(inlineIntNull(ret_ti));
    datediff_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      datediff_fname, get_int_type(64, cgen_state_->context_), datediff_args);
}

llvm::Value* Executor::codegen(const Analyzer::DatetruncExpr* datetrunc_expr,
                               const CompilationOptions& co) {
  auto from_expr = codegen(datetrunc_expr->get_from_expr(), true, co).front();
  const auto& datetrunc_expr_ti = datetrunc_expr->get_from_expr()->get_type_info();
  CHECK(from_expr->getType()->isIntegerTy(32) || from_expr->getType()->isIntegerTy(64));
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  if (sizeof(time_t) == 4 && from_expr->getType()->isIntegerTy(64)) {
    from_expr =
        cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::Trunc,
                                            from_expr,
                                            get_int_type(32, cgen_state_->context_));
  }
  std::vector<llvm::Value*> datetrunc_args{
      ll_int(static_cast<int32_t>(datetrunc_expr->get_field())), from_expr};
  std::string datetrunc_fname{"DateTruncate"};
  if (datetrunc_expr_ti.get_dimension() > 0) {
    datetrunc_args.push_back(ll_int(static_cast<int64_t>(
        get_timestamp_precision_scale(datetrunc_expr_ti.get_dimension()))));
    datetrunc_fname += "HighPrecision";
  }
  if (!datetrunc_expr_ti.get_notnull()) {
    datetrunc_args.push_back(inlineIntNull(datetrunc_expr_ti));
    datetrunc_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      datetrunc_fname, get_int_type(64, cgen_state_->context_), datetrunc_args);
}
