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

#include "WindowContext.h"
#include <numeric>
#include "../Shared/checked_alloc.h"

WindowFunctionContext::WindowFunctionContext(
    const Analyzer::WindowFunction* window_func,
    const std::shared_ptr<JoinHashTableInterface>& partitions,
    const size_t elem_count,
    const ExecutorDeviceType device_type)
    : window_func_(window_func)
    , partitions_(partitions)
    , elem_count_(elem_count)
    , output_(nullptr)
    , device_type_(device_type) {}

WindowFunctionContext::~WindowFunctionContext() {
  for (auto order_column_partitioned : order_columns_partitioned_) {
    free(order_column_partitioned);
  }
  free(output_);
}

void WindowFunctionContext::addOrderColumn(const int8_t* column,
                                           const Analyzer::ColumnVar* col_var) {
  const auto& col_ti = col_var->get_type_info();
  auto partitioned_dest =
      static_cast<int8_t*>(checked_malloc(elem_count_ * col_ti.get_size()));
  order_columns_partitioned_.push_back(partitioned_dest);
  switch (col_ti.get_size()) {
    case 8: {
      scatterToPartitions(reinterpret_cast<int64_t*>(partitioned_dest),
                          reinterpret_cast<const int64_t*>(column),
                          payload(),
                          elem_count_);
      break;
    }
    case 4: {
      scatterToPartitions(reinterpret_cast<int32_t*>(partitioned_dest),
                          reinterpret_cast<const int32_t*>(column),
                          payload(),
                          elem_count_);
      break;
    }
    case 2: {
      scatterToPartitions(reinterpret_cast<int16_t*>(partitioned_dest),
                          reinterpret_cast<const int16_t*>(column),
                          payload(),
                          elem_count_);
      break;
    }
    case 1: {
      scatterToPartitions(partitioned_dest, column, payload(), elem_count_);
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported size: " << col_ti.get_size();
      break;
    }
  }
}

namespace {

std::vector<int64_t> index_to_rank(const int64_t* index, const size_t index_size) {
  std::vector<int64_t> rank(index_size);
  for (size_t i = 0; i < index_size; ++i) {
    rank[index[i]] = i + 1;
  }
  return rank;
}

}  // namespace

void WindowFunctionContext::compute() {
  CHECK(!output_);
  const auto& ti = window_func_->get_type_info();
  output_ = static_cast<int8_t*>(checked_malloc(elem_count_ * ti.get_logical_size()));
  switch (window_func_->getKind()) {
    case SqlWindowFunctionKind::ROW_NUMBER: {
      std::unique_ptr<int64_t[]> scratchpad(new int64_t[elem_count_]);
      CHECK_EQ(ti.get_size(), size_t(8));
      const auto partition_count = counts() - offsets();
      CHECK_GE(partition_count, 0);
      for (size_t i = 0; i < static_cast<size_t>(partition_count); ++i) {
        auto partition_size = counts()[i];
        auto output_for_partition_buff = scratchpad.get() + offsets()[i];
        std::iota(output_for_partition_buff,
                  output_for_partition_buff + partition_size,
                  int64_t(0));
        for (size_t order_column_idx = 0;
             order_column_idx < order_columns_partitioned_.size();
             ++order_column_idx) {
          auto order_column_partitioned = order_columns_partitioned_[order_column_idx];
          const auto& order_keys = window_func_->getOrderKeys();
          const auto order_col = dynamic_cast<const Analyzer::ColumnVar*>(
              order_keys[order_column_idx].get());
          CHECK(order_col);
          const auto& order_col_ti = order_col->get_type_info();
          auto order_column_partition =
              order_column_partitioned + offsets()[i] * order_col_ti.get_size();
          std::stable_sort(output_for_partition_buff,
                           output_for_partition_buff + partition_size,
                           makeComparator(order_col, order_column_partition));
          const auto rank = index_to_rank(output_for_partition_buff, partition_size);
          std::copy(rank.begin(), rank.end(), output_for_partition_buff);
        }
      }
      auto output_i64 = reinterpret_cast<int64_t*>(output_);
      for (size_t i = 0; i < elem_count_; ++i) {
        output_i64[payload()[i]] = scratchpad[i];
      }
      break;
    }
    default: { LOG(FATAL) << "Invalid window function kind"; }
  }
}

const Analyzer::WindowFunction* WindowFunctionContext::getWindowFunction() const {
  return window_func_;
}

const int8_t* WindowFunctionContext::output() const {
  return output_;
}

std::function<bool(const int64_t lhs, const int64_t rhs)>
WindowFunctionContext::makeComparator(const Analyzer::ColumnVar* col_var,
                                      const int8_t* partition_values) {
  const auto& ti = col_var->get_type_info();
  switch (ti.get_type()) {
    case kBIGINT: {
      const auto values = reinterpret_cast<const int64_t*>(partition_values);
      return [values](const int64_t lhs, const int64_t rhs) {
        return values[lhs] < values[rhs];
      };
    }
    case kINT: {
      const auto values = reinterpret_cast<const int32_t*>(partition_values);
      return [values](const int64_t lhs, const int64_t rhs) {
        return values[lhs] < values[rhs];
      };
    }
    case kSMALLINT: {
      const auto values = reinterpret_cast<const int16_t*>(partition_values);
      return [values](const int64_t lhs, const int64_t rhs) {
        return values[lhs] < values[rhs];
      };
    }
    case kTINYINT: {
      return [partition_values](const int64_t lhs, const int64_t rhs) {
        return partition_values[lhs] < partition_values[rhs];
      };
    }
    case kFLOAT: {
      const auto values = reinterpret_cast<const float*>(partition_values);
      return [values](const int64_t lhs, const int64_t rhs) {
        return values[lhs] < values[rhs];
      };
    }
    case kDOUBLE: {
      const auto values = reinterpret_cast<const double*>(partition_values);
      return [values](const int64_t lhs, const int64_t rhs) {
        return values[lhs] < values[rhs];
      };
    }
    default: { LOG(FATAL) << "Type not supported yet"; }
  }
  return nullptr;
}

template <class T>
void WindowFunctionContext::scatterToPartitions(T* dest,
                                                const T* source,
                                                const int32_t* positions,
                                                const size_t elem_count) {
  for (size_t i = 0; i < elem_count; ++i) {
    dest[i] = source[positions[i]];
  }
}

const int32_t* WindowFunctionContext::payload() const {
  return reinterpret_cast<const int32_t*>(
      partitions_->getJoinHashBuffer(device_type_, 0) + partitions_->payloadBufferOff());
}

const int32_t* WindowFunctionContext::offsets() const {
  return reinterpret_cast<const int32_t*>(
      partitions_->getJoinHashBuffer(device_type_, 0) + partitions_->offsetBufferOff());
}

const int32_t* WindowFunctionContext::counts() const {
  return reinterpret_cast<const int32_t*>(
      partitions_->getJoinHashBuffer(device_type_, 0) + partitions_->countBufferOff());
}

void WindowProjectNodeContext::addWindowFunctionContext(
    std::unique_ptr<WindowFunctionContext> window_function_context,
    const size_t target_index) {
  const auto it_ok = window_contexts_.emplace(
      std::make_pair(target_index, std::move(window_function_context)));
  CHECK(it_ok.second);
}

const WindowFunctionContext* WindowProjectNodeContext::getWindowFunctionContext(
    const size_t target_index) const {
  const auto it = window_contexts_.find(target_index);
  CHECK(it != window_contexts_.end());
  return it->second.get();
}

WindowProjectNodeContext* WindowProjectNodeContext::reset() {
  s_instance_ = std::make_unique<WindowProjectNodeContext>();
  return s_instance_.get();
}

const WindowProjectNodeContext* WindowProjectNodeContext::get() {
  return s_instance_.get();
}

std::unique_ptr<WindowProjectNodeContext> WindowProjectNodeContext::s_instance_;
