/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/AggregationContext.h"

#include <algorithm>

#include "QueryRenderer/Data/QuerySourceDataTable.h"
#include "QueryRenderer/Marks/BaseMark.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/VegaElements.h"

namespace QueryRenderer {

AggregationContext::AggregationContext(QueryRendererContext& render_context)
    : render_context_(render_context) {}

namespace {
bool is_agg_renderable_mark(const BaseQueryDataTableSQLJSON* query_data) {
  if (!query_data || (query_data->getResultSet() && query_data->isNonInSitu())) {
    return true;
  }
  return false;
}
}  // namespace

size_t AggregationContext::getNumAggRenderableMarks() const {
  auto const& mark_vector = render_context_.getVegaElements().getMarkVector();
  return std::count_if(
      mark_vector.begin(), mark_vector.end(), [](const BaseMarkUqPtr& mark) {
        auto data_table = mark->getDataPtr();
        if (data_table) {
          auto query_data =
              std::dynamic_pointer_cast<const BaseQueryDataTableSQLJSON>(data_table);
          return is_agg_renderable_mark(query_data.get());
        } else {
          return true;
        }
        return false;
      });
}

size_t AggregationContext::getNumSoloAggRenderableMarks() const {
  auto const& mark_vector = render_context_.getVegaElements().getMarkVector();
  return std::count_if(
      mark_vector.begin(), mark_vector.end(), [](const BaseMarkUqPtr& mark) {
        const auto data_refs = mark->getDataRefs();
        for (const auto& data_ref : data_refs) {
          auto* query_data = dynamic_cast<BaseQueryDataTableSQLJSON*>(data_ref.get());
          if (!is_agg_renderable_mark(query_data)) {
            return false;
          } else if (!query_data) {
            auto* query_source_data = dynamic_cast<QuerySourceDataTable*>(data_ref.get());
            if (query_source_data) {
              auto source = query_source_data->getSourceDataRef();
              query_data = dynamic_cast<BaseQueryDataTableSQLJSON*>(source.get());
              if (!is_agg_renderable_mark(query_data)) {
                return false;
              }
            }
          }
        }
        return true;
      });
}

}  // namespace QueryRenderer
