/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include "QueryEngine/GeoOperators/Codegen.h"

namespace spatial_type {

class NumGeometries : public Codegen {
 public:
  NumGeometries(const Analyzer::GeoOperator* geo_operator,
                const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog) {}

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kINT); }

  const SQLTypeInfo getOperandTypeInfo(const size_t index) {
    CHECK_EQ(index, size_t(0));
    const auto operand = operator_->getOperand(0);
    auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(operand);
    if (!col_var) {
      throw std::runtime_error(getName() +
                               " requires a geo column as its input argument.");
    }
    return col_var->get_type_info();
  }

  const Analyzer::Expr* getOperand(const size_t index) final {
    CHECK_EQ(index, size_t(0));
    if (operand_owned_) {
      return operand_owned_.get();
    }

    const auto operand = operator_->getOperand(0);
    auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(operand);
    if (!col_var) {
      throw std::runtime_error(getName() +
                               " requires a geo column as its input argument.");
    }

    const auto& geo_ti = col_var->get_type_info();
    if (!geo_ti.is_geometry()) {
      throw std::runtime_error(getName() +
                               " requires a geo column as its input argument.");
    }
    is_nullable_ = !geo_ti.get_notnull();

    auto const geo_type = geo_ti.get_type();
    int column_offset{};
    switch (geo_type) {
      case kMULTIPOLYGON:
        column_offset = 3;  // poly_rings
        break;
      case kMULTILINESTRING:
        column_offset = 2;  // ring_sizes
        break;
      case kMULTIPOINT:
        column_offset = 1;  // points
        break;
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
        column_offset = 1;  // nothing to count, but allow through
        break;
      default:
        UNREACHABLE();
    }

    // create a new operand which is just the column to count, and codegen it
    const auto column_id = col_var->get_column_id() + column_offset;
    auto cd = get_column_descriptor(column_id, col_var->get_table_id(), *cat_);
    CHECK(cd);

    operand_owned_ = std::make_unique<Analyzer::ColumnVar>(
        cd->columnType, col_var->get_table_id(), column_id, col_var->get_rte_idx());
    return operand_owned_.get();
  }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    CHECK_EQ(pos_lvs.size(), size());
    CHECK_EQ(arg_lvs.size(), size_t(1));

    const auto& oper_ti = getOperandTypeInfo(0);
    auto const is_multi_geo = IS_GEO_MULTI(oper_ti.get_type());

    // non-MULTI non-nullable geo, just return 1 (no function call)
    if (!isNullable() && !is_multi_geo) {
      auto* const one = cgen_state->llInt(1);
      return {{one}, one};
    }

    // will be a function call, the first two arguments
    std::string fn_name("array_size");
    auto& argument_lv = arg_lvs.front();
    std::vector<llvm::Value*> array_size_args = {argument_lv, pos_lvs.front()};

    if (is_multi_geo) {
      // MULTI geo, fetch and append element log size argument
      const auto& elem_ti = getOperand(0)->get_type_info().get_elem_type();
      uint32_t elem_log_sz_value{};
      if (oper_ti.get_type() == kMULTIPOINT) {
        // we must have been passed the coords column
        CHECK(elem_ti.get_type() == kTINYINT);
        // we want to return the number of points (two coords),
        // so divide by either 8 (compressed) or 16 (uncompressed)
        if (oper_ti.get_compression() == kENCODING_GEOINT) {
          // number of INT pairs
          elem_log_sz_value = 3;
        } else {
          // number of DOUBLE pairs
          elem_log_sz_value = 4;
        }
      } else {
        // some other count (ring_sizes or poly_sizes)
        elem_log_sz_value = log2_bytes(elem_ti.get_logical_size());
      }
      array_size_args.push_back(cgen_state->llInt(elem_log_sz_value));
    } else {
      // non-MULTI but nullable geo, return 1 or NULL
      fn_name += "_1";
    }

    // nullable, add NULL value
    if (isNullable()) {
      fn_name += "_nullable";
      array_size_args.push_back(cgen_state->inlineIntNull(getTypeInfo()));
    }

    const auto total_num_geometries_lv = cgen_state->emitExternalCall(
        fn_name, get_int_type(32, cgen_state->context_), array_size_args);

    return {{total_num_geometries_lv}, total_num_geometries_lv};
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    CHECK_EQ(args.size(), size_t(1));
    if (isNullable()) {
      CHECK(nullcheck_codegen);
      return {nullcheck_codegen->finalize(cgen_state->inlineIntNull(getTypeInfo()),
                                          args.front())};
    }
    return {args.front()};
  }

 protected:
  std::unique_ptr<Analyzer::ColumnVar> operand_owned_;
};

}  // namespace spatial_type
