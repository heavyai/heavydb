/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// we do not reorder tables when they are used as input of ST_CONTAINS function
// to prevent incorrect query result
#pragma once

#include <vector>

#include "Analyzer/Analyzer.h"
#include "Shared/DbObjectKeys.h"

struct GeoJoinOperandsTableKeyPair {
  shared::TableKey inner_table_key;
  shared::TableKey outer_table_key;
};

class GeospatialFunctionFinder : public ScalarExprVisitor<void*> {
 public:
  const std::vector<const Analyzer::ColumnVar*>& getGeoArgCvs() const;
  const std::optional<GeoJoinOperandsTableKeyPair> getJoinTableKeyPair() const;
  const std::string& getGeoFunctionName() const;

 protected:
  void* visitGeoExpr(const Analyzer::GeoExpr* geo_expr) const override;
  void* visitFunctionOper(const Analyzer::FunctionOper* func_oper) const override;

 private:
  mutable std::vector<const Analyzer::ColumnVar*> geo_arg_cvs_;
  mutable std::string geo_func_name_{""};
};
