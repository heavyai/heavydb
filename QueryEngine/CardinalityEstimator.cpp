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

#include "CardinalityEstimator.h"
#include "ErrorHandling.h"
#include "ExpressionRewrite.h"
#include "RelAlgExecutor.h"

int64_t g_large_ndv_threshold = 10000000;
size_t g_large_ndv_multiplier = 256;

namespace Analyzer {

size_t LargeNDVEstimator::getBufferSize() const {
  return 1024 * 1024 * g_large_ndv_multiplier;
}

}  // namespace Analyzer

size_t ResultSet::getNDVEstimator() const {
  CHECK(dynamic_cast<const Analyzer::NDVEstimator*>(estimator_.get()));
  CHECK(host_estimator_buffer_);
  auto bits_set = bitmap_set_size(host_estimator_buffer_, estimator_->getBufferSize());
  if (bits_set == 0) {
    // empty result set, return 1 for a groups buffer size of 1
    return 1;
  }
  const auto total_bits = estimator_->getBufferSize() * 8;
  CHECK_LE(bits_set, total_bits);
  const auto unset_bits = total_bits - bits_set;
  const auto ratio = static_cast<double>(unset_bits) / total_bits;
  if (ratio == 0.) {
    LOG(WARNING)
        << "Failed to get a high quality cardinality estimation, falling back to "
           "approximate group by buffer size guess.";
    return 0;
  }
  return -static_cast<double>(total_bits) * log(ratio);
}

size_t RelAlgExecutor::getNDVEstimation(const WorkUnit& work_unit,
                                        const int64_t range,
                                        const bool is_agg,
                                        const CompilationOptions& co,
                                        const ExecutionOptions& eo) {
  const auto estimator_exe_unit = work_unit.exe_unit.createNdvExecutionUnit(range);
  size_t one{1};
  ColumnCacheMap column_cache;
  try {
    const auto estimator_result =
        executor_->executeWorkUnit(one,
                                   is_agg,
                                   get_table_infos(work_unit.exe_unit, executor_),
                                   estimator_exe_unit,
                                   co,
                                   eo,
                                   nullptr,
                                   false,
                                   column_cache);
    if (!estimator_result) {
      return 1;  // empty row set, only needs one slot
    }
    return estimator_result->getNDVEstimator();
  } catch (const QueryExecutionError& e) {
    if (e.hasErrorCode(ErrorCode::OUT_OF_TIME)) {
      throw std::runtime_error("Cardinality estimation query ran out of time");
    }
    if (e.hasErrorCode(ErrorCode::INTERRUPTED)) {
      throw std::runtime_error("Cardinality estimation query has been interrupted");
    }
    throw std::runtime_error("Failed to run the cardinality estimation query: " +
                             getErrorMessageFromCode(e.getErrorCode()));
  }
  UNREACHABLE();
  return 0;
}

RelAlgExecutionUnit RelAlgExecutionUnit::createNdvExecutionUnit(
    const int64_t range) const {
  const bool use_large_estimator =
      range > g_large_ndv_threshold || groupby_exprs.size() > 1;
  return {input_descs,
          input_col_descs,
          simple_quals,
          quals,
          join_quals,
          {},
          {},
          {},
          use_large_estimator ? makeExpr<Analyzer::LargeNDVEstimator>(groupby_exprs)
                              : makeExpr<Analyzer::NDVEstimator>(groupby_exprs),
          SortInfo(),
          0,
          query_hint,
          query_plan_dag_hash,
          hash_table_build_plan_dag,
          table_id_to_node_map,
          false,
          union_all,
          query_state,
          {}};
}

RelAlgExecutionUnit RelAlgExecutionUnit::createCountAllExecutionUnit(
    Analyzer::Expr* replacement_target) const {
  return {input_descs,
          input_col_descs,
          simple_quals,
          strip_join_covered_filter_quals(quals, join_quals),
          join_quals,
          {},
          {replacement_target},
          {},
          nullptr,
          SortInfo(),
          0,
          query_hint,
          query_plan_dag_hash,
          hash_table_build_plan_dag,
          table_id_to_node_map,
          false,
          union_all,
          query_state,
          {replacement_target},
          /*per_device_cardinality=*/{}};
}

ResultSetPtr reduce_estimator_results(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device) {
  if (results_per_device.empty()) {
    return nullptr;
  }
  CHECK(dynamic_cast<const Analyzer::NDVEstimator*>(ra_exe_unit.estimator.get()));
  const auto& result_set = results_per_device.front().first;
  CHECK(result_set);
  auto estimator_buffer = result_set->getHostEstimatorBuffer();
  CHECK(estimator_buffer);
  for (size_t i = 1; i < results_per_device.size(); ++i) {
    const auto& next_result_set = results_per_device[i].first;
    const auto other_estimator_buffer = next_result_set->getHostEstimatorBuffer();
    for (size_t off = 0; off < ra_exe_unit.estimator->getBufferSize(); ++off) {
      estimator_buffer[off] |= other_estimator_buffer[off];
    }
  }
  return std::move(result_set);
}
