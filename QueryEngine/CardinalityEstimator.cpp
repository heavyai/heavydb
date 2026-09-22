/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CardinalityEstimator.h"
#include "ErrorHandling.h"
#include "ExpressionRewrite.h"
#include "HyperLogLog.h"
#include "RelAlgExecutor.h"
#include "Shared/threading.h"

#include <algorithm>

int64_t g_large_ndv_threshold = 10000000;
size_t g_large_ndv_multiplier = 256;
extern bool g_enable_result_reduction_pipeline;

namespace Analyzer {

size_t LargeNDVEstimator::getBufferSize() const {
  return 1024 * 1024 * g_large_ndv_multiplier;
}

}  // namespace Analyzer

size_t ResultSet::getNDVEstimator() const {
  CHECK(dynamic_cast<const Analyzer::NDVEstimator*>(estimator_.get()));
  CHECK(host_estimator_buffer_);
  if (const auto hll_estimator =
          dynamic_cast<const Analyzer::HllNDVEstimator*>(estimator_.get())) {
    // Preserve the one-slot empty-input contract used by the linear estimator.
    return std::max(size_t{1},
                    hll_size(reinterpret_cast<const int32_t*>(host_estimator_buffer_),
                             hll_estimator->kPrecisionBits));
  }
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
    if (e.hasErrorCode(ErrorCode::OUT_OF_GPU_MEM)) {
      LOG(WARNING) << "Cardinality estimation query ran out of GPU memory; falling "
                      "back to approximate group-by buffer sizing.";
      return 0;
    }
    if (e.hasErrorCode(ErrorCode::OVERFLOW_OR_UNDERFLOW)) {
      LOG(WARNING) << "Cardinality estimation query overflowed; falling back to "
                      "approximate group-by buffer sizing.";
      return 0;
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
  const auto estimator =
      g_enable_result_reduction_pipeline
          ? std::static_pointer_cast<Analyzer::Estimator>(
                makeExpr<Analyzer::HllNDVEstimator>(groupby_exprs))
          : (use_large_estimator
                 ? std::static_pointer_cast<Analyzer::Estimator>(
                       makeExpr<Analyzer::LargeNDVEstimator>(groupby_exprs))
                 : std::static_pointer_cast<Analyzer::Estimator>(
                       makeExpr<Analyzer::NDVEstimator>(groupby_exprs)));
  return {input_descs,
          input_col_descs,
          simple_quals,
          quals,
          join_quals,
          {},
          {},
          estimator,
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
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
    const size_t executor_id) {
  if (results_per_device.empty()) {
    return nullptr;
  }
  CHECK(dynamic_cast<const Analyzer::NDVEstimator*>(ra_exe_unit.estimator.get()));
  const auto& result_set = results_per_device.front().first;
  CHECK(result_set);
  auto estimator_buffer = result_set->getHostEstimatorBuffer();
  CHECK(estimator_buffer);
  auto executor = Executor::getExecutor(executor_id);
  CHECK(executor);
  const auto check_interrupted = [&executor] {
    if (UNLIKELY(executor->checkNonKernelTimeInterrupted())) {
      throw QueryExecutionError(ErrorCode::INTERRUPTED);
    }
  };
  const auto buffer_size = ra_exe_unit.estimator->getBufferSize();
  CHECK_GT(buffer_size, size_t(0));
  std::vector<const int8_t*> source_buffers;
  source_buffers.reserve(results_per_device.size() - 1);
  for (size_t i = 1; i < results_per_device.size(); ++i) {
    const auto& next_result_set = results_per_device[i].first;
    const auto other_estimator_buffer = next_result_set->getHostEstimatorBuffer();
    CHECK(other_estimator_buffer);
    source_buffers.push_back(other_estimator_buffer);
  }
  if (source_buffers.empty()) {
    return std::move(result_set);
  }

  using namespace threading;
  if (const auto hll_estimator =
          dynamic_cast<const Analyzer::HllNDVEstimator*>(ra_exe_unit.estimator.get())) {
    const auto register_count = size_t{1} << hll_estimator->kPrecisionBits;
    CHECK_EQ(buffer_size, register_count * sizeof(int32_t));
    auto estimator_registers = reinterpret_cast<int32_t*>(estimator_buffer);
    std::vector<const int32_t*> source_registers;
    source_registers.reserve(source_buffers.size());
    for (const auto source_buffer : source_buffers) {
      source_registers.push_back(reinterpret_cast<const int32_t*>(source_buffer));
    }
    constexpr size_t kEstimatorReductionGrainRegisters = 16 * 1024;
    parallel_for(
        blocked_range<size_t>(0, register_count, kEstimatorReductionGrainRegisters),
        [&](const blocked_range<size_t>& range) {
          for (const auto source_register_buffer : source_registers) {
            check_interrupted();
            for (size_t off = range.begin(); off < range.end(); ++off) {
              estimator_registers[off] =
                  std::max(estimator_registers[off], source_register_buffer[off]);
            }
          }
        });
    return std::move(result_set);
  }

  const auto word_count = buffer_size / sizeof(uint64_t);
  auto estimator_words = reinterpret_cast<uint64_t*>(estimator_buffer);
  std::vector<const uint64_t*> source_words;
  source_words.reserve(source_buffers.size());
  for (const auto source_buffer : source_buffers) {
    source_words.push_back(reinterpret_cast<const uint64_t*>(source_buffer));
  }
  constexpr size_t kEstimatorReductionGrainWords = 64 * 1024;
  parallel_for(blocked_range<size_t>(0, word_count, kEstimatorReductionGrainWords),
               [&](const blocked_range<size_t>& range) {
                 for (const auto source_word_buffer : source_words) {
                   check_interrupted();
                   for (size_t off = range.begin(); off < range.end(); ++off) {
                     estimator_words[off] |= source_word_buffer[off];
                   }
                 }
               });

  const auto tail_begin = word_count * sizeof(uint64_t);
  if (tail_begin < buffer_size) {
    check_interrupted();
    for (const auto source_buffer : source_buffers) {
      for (size_t off = tail_begin; off < buffer_size; ++off) {
        estimator_buffer[off] |= source_buffer[off];
      }
    }
  }
  return std::move(result_set);
}
