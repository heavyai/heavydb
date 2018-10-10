/*
 * Copyright 2018 OmniSci, Inc.
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
#include "RelAlgExecutor.h"

size_t ResultSet::getNDVEstimator() const {
  CHECK(dynamic_cast<const Analyzer::NDVEstimator*>(estimator_.get()));
  CHECK(host_estimator_buffer_);
  auto bits_set = bitmap_set_size(host_estimator_buffer_, estimator_->getBufferSize());
  const auto total_bits = estimator_->getBufferSize() * 8;
  CHECK_LE(bits_set, total_bits);
  const auto unset_bits = total_bits - bits_set;
  const auto ratio = static_cast<double>(unset_bits) / total_bits;
  if (ratio == 0.) {
    throw std::runtime_error("Failed to get a high quality cardinality estimation");
  }
  return -static_cast<double>(total_bits) * log(ratio);
}

size_t RelAlgExecutor::getNDVEstimation(const WorkUnit& work_unit,
                                        const bool is_agg,
                                        const CompilationOptions& co,
                                        const ExecutionOptions& eo) {
  const auto estimator_exe_unit = create_ndv_execution_unit(work_unit.exe_unit);
  int32_t error_code{0};
  size_t one{1};
  const auto estimator_result =
      executor_->executeWorkUnit(&error_code,
                                 one,
                                 is_agg,
                                 get_table_infos(work_unit.exe_unit, executor_),
                                 estimator_exe_unit,
                                 co,
                                 eo,
                                 cat_,
                                 executor_->row_set_mem_owner_,
                                 nullptr,
                                 false);
  if (error_code == Executor::ERR_OUT_OF_TIME) {
    throw std::runtime_error("Cardinality estimation query ran out of time");
  }
  if (error_code == Executor::ERR_INTERRUPTED) {
    throw std::runtime_error("Cardinality estimation query has been interrupted");
  }
  if (error_code) {
    throw std::runtime_error("Failed to run the cardinality estimation query: " +
                             getErrorMessageFromCode(error_code));
  }
  const auto& estimator_result_rows = boost::get<RowSetPtr>(estimator_result);
  if (!estimator_result_rows) {
    return 1;
  }
  return std::max(estimator_result_rows->getNDVEstimator(), size_t(1));
}

RelAlgExecutionUnit create_ndv_execution_unit(const RelAlgExecutionUnit& ra_exe_unit) {
  return {ra_exe_unit.input_descs,
          ra_exe_unit.extra_input_descs,
          ra_exe_unit.input_col_descs,
          ra_exe_unit.simple_quals,
          ra_exe_unit.quals,
          ra_exe_unit.join_type,
          ra_exe_unit.inner_joins,
          ra_exe_unit.join_dimensions,
          ra_exe_unit.inner_join_quals,
          ra_exe_unit.outer_join_quals,
          {},
          {},
          makeExpr<Analyzer::NDVEstimator>(ra_exe_unit.groupby_exprs),
          SortInfo{{}, SortAlgorithm::Default, 0, 0},
          0};
}

RelAlgExecutionUnit create_count_all_execution_unit(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::shared_ptr<Analyzer::Expr> replacement_target) {
  return {ra_exe_unit.input_descs,
          ra_exe_unit.extra_input_descs,
          ra_exe_unit.input_col_descs,
          ra_exe_unit.simple_quals,
          ra_exe_unit.quals,
          ra_exe_unit.join_type,
          ra_exe_unit.inner_joins,
          ra_exe_unit.join_dimensions,
          ra_exe_unit.inner_join_quals,
          ra_exe_unit.outer_join_quals,
          {},
          {replacement_target.get()},
          nullptr,
          SortInfo{{}, SortAlgorithm::Default, 0, 0},
          0};
}

RowSetPtr reduce_estimator_results(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultPtr, std::vector<size_t>>>& results_per_device) {
  if (results_per_device.empty()) {
    return nullptr;
  }
  CHECK(dynamic_cast<const Analyzer::NDVEstimator*>(ra_exe_unit.estimator.get()));
  auto first = boost::get<RowSetPtr>(&results_per_device.front().first);
  CHECK(first && *first);
  const auto& result_set = *first;
  CHECK(result_set);
  auto estimator_buffer = result_set->getHostEstimatorBuffer();
  CHECK(estimator_buffer);
  for (size_t i = 1; i < results_per_device.size(); ++i) {
    auto next = boost::get<RowSetPtr>(&results_per_device[i].first);
    CHECK(next && *next);
    const auto& next_result_set = *next;
    const auto other_estimator_buffer = next_result_set->getHostEstimatorBuffer();
    for (size_t off = 0; off < ra_exe_unit.estimator->getBufferSize(); ++off) {
      estimator_buffer[off] |= other_estimator_buffer[off];
    }
  }
  return std::move(*first);
}
