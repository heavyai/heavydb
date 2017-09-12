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

#include "RelAlgTranslator.h"

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateGeoFunction(const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto rex_input0 = dynamic_cast<const RexInput*>(rex_function->getOperand(0));
  const auto rex_input1 = dynamic_cast<const RexInput*>(rex_function->getOperand(1));
  if (!rex_input0 || !rex_input1) {
    throw QueryNotSupported("Function " + rex_function->getName() + " not supported for arguments provided");
  }
  const auto input0 = translateInput(rex_input0);
  const auto input1 = translateInput(rex_input1);
  const auto columnA = dynamic_cast<const Analyzer::ColumnVar*>(input0.get());
  const auto columnB = dynamic_cast<const Analyzer::ColumnVar*>(input1.get());
  if (!columnA || !columnB) {
    throw QueryNotSupported("Function " + rex_function->getName() + " not supported for arguments provided");
  }
  const auto& columnA_ti = columnA->get_type_info();
  const auto& columnB_ti = columnB->get_type_info();
  if (rex_function->getName() == std::string("ST_Distance")) {
    if (columnA_ti.get_type() == kPOINT) {
      if (columnB_ti.get_type() == kPOINT) {
        return makeExpr<Analyzer::FunctionOper>(
            rex_function->getType(), "ST_Distance_Point_Point", translateGeoFunctionArgs(rex_function));
      }
      if (columnB_ti.get_type() == kLINE) {
        return makeExpr<Analyzer::FunctionOper>(
            rex_function->getType(), "ST_Distance_Point_Line", translateGeoFunctionArgs(rex_function));
      }
    }
    if (columnA_ti.get_type() == kLINE) {
      if (columnB_ti.get_type() == kPOINT) {
        return makeExpr<Analyzer::FunctionOper>(
            rex_function->getType(), "ST_Distance_Line_Point", translateGeoFunctionArgs(rex_function));
      }
      if (columnB_ti.get_type() == kLINE) {
        return makeExpr<Analyzer::FunctionOper>(
            rex_function->getType(), "ST_Distance_Line_Line", translateGeoFunctionArgs(rex_function));
      }
    }
    throw QueryNotSupported("Function " + rex_function->getName() + " not supported for arguments provided");
  }
  if (rex_function->getName() == std::string("ST_Contains")) {
    if (columnA_ti.get_type() == kPOINT) {
      if (columnB_ti.get_type() == kPOINT) {
        return makeExpr<Analyzer::FunctionOper>(
            rex_function->getType(), "ST_Contains_Point_Point", translateGeoFunctionArgs(rex_function));
      }
      if (columnB_ti.get_type() == kLINE) {
        return makeExpr<Analyzer::FunctionOper>(
            rex_function->getType(), "ST_Contains_Point_Line", translateGeoFunctionArgs(rex_function));
      }
    }
    if (columnA_ti.get_type() == kLINE) {
      if (columnB_ti.get_type() == kPOINT) {
        return makeExpr<Analyzer::FunctionOper>(
            rex_function->getType(), "ST_Contains_Line_Point", translateGeoFunctionArgs(rex_function));
      }
      if (columnB_ti.get_type() == kLINE) {
        return makeExpr<Analyzer::FunctionOper>(
            rex_function->getType(), "ST_Contains_Line_Line", translateGeoFunctionArgs(rex_function));
      }
    }
    throw QueryNotSupported("Function " + rex_function->getName() + " not supported for arguments provided");
  }

  throw QueryNotSupported("Function " + rex_function->getName() + " not supported");
  return nullptr;
}

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoFunctionArgs(
    const RexFunctionOperator* rex_function) const {
  // Translate geo columns references to a list of physical column refs, based on geo type
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  for (size_t i = 0; i < rex_function->size(); ++i) {
    const auto rex_input = dynamic_cast<const RexInput*>(rex_function->getOperand(i));
    auto columns = translateGeoFunctionArg(rex_input);
    for (auto column : columns) {
      args.push_back(column);
    }
  }
  return args;
}

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateGeoFunctionArg(
    const RexInput* rex_input) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  const auto source = rex_input->getSourceNode();
  const auto it_rte_idx = input_to_nest_level_.find(source);
  CHECK(it_rte_idx != input_to_nest_level_.end());
  const int rte_idx = it_rte_idx->second;
  const auto scan_source = dynamic_cast<const RelScan*>(source);
  const auto& in_metainfo = source->getOutputMetainfo();
  CHECK(scan_source);
  // We're at leaf (scan) level and not supposed to have input metadata,
  // the name and type information come directly from the catalog.
  CHECK(in_metainfo.empty());
  const auto table_desc = scan_source->getTableDescriptor();
  const auto cd = cat_.getMetadataForColumn(table_desc->tableId, rex_input->getIndex() + 1);
  CHECK(cd);
  CHECK(!cd->isPhysicalCol);
  CHECK_GT(cd->numPhysicalColumns, 0);
  auto col_ti = cd->columnType;
  CHECK(IS_GEO(col_ti.get_type()));
  for (auto i = 0; i < cd->numPhysicalColumns; i++) {
    const auto cd0 = cat_.getMetadataForColumn(table_desc->tableId, rex_input->getIndex() + 1 + i + 1);
    auto col0_ti = cd0->columnType;
    CHECK(cd0->isPhysicalCol);
    CHECK(!cd0->isVirtualCol);
    args.push_back(std::make_shared<Analyzer::ColumnVar>(col0_ti, table_desc->tableId, cd0->columnId, rte_idx));
  }
  return args;
}
