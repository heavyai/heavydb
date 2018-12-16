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

#pragma once

#include "../Analyzer/Analyzer.h"
#include "JoinHashTableInterface.h"

// Per-window function context which encapsulates the logic for computing the various
// window function kinds and keeps ownership of buffers which contain the results. For
// rank functions, the code generated for the projection simply reads the values and
// writes them to the result set. For value and aggregate functions, only the iteration
// order is written to the buffer, the rest is handled by generating code in a similar way
// we do for non-window queries.
class WindowFunctionContext {
 public:
  WindowFunctionContext(const Analyzer::WindowFunction* window_func,
                        const std::shared_ptr<JoinHashTableInterface>& partitions,
                        const size_t elem_count,
                        const ExecutorDeviceType device_type);

  WindowFunctionContext(const WindowFunctionContext&) = delete;

  WindowFunctionContext& operator=(const WindowFunctionContext&) = delete;

  ~WindowFunctionContext();

  void addOrderColumn(const int8_t* column, const Analyzer::ColumnVar* col_var);

  void compute();

  const Analyzer::WindowFunction* getWindowFunction() const;

  const int8_t* output() const;

  using Comparator = std::function<bool(const int64_t lhs, const int64_t rhs)>;

 private:
  static Comparator makeComparator(const Analyzer::ColumnVar* col_var,
                                   const int8_t* partition_values);

  template <class T>
  static void scatterToPartitions(T* dest,
                                  const T* source,
                                  const int32_t* positions,
                                  const size_t elem_count);

  static void computePartition(int64_t* output_for_partition_buff,
                               const size_t partition_size,
                               const size_t off,
                               const Analyzer::WindowFunction* window_func,
                               const std::vector<Comparator>& comparators);

  const int32_t* payload() const;

  const int32_t* offsets() const;

  const int32_t* counts() const;

  const Analyzer::WindowFunction* window_func_;
  // The order column buffers partitioned as specified by partitions_.
  std::vector<int8_t*> order_columns_partitioned_;
  // Hash table which contains the partitions specified by the window.
  std::shared_ptr<JoinHashTableInterface> partitions_;
  // The number of elements in the table.
  size_t elem_count_;
  // The output of the window function.
  int8_t* output_;
  const ExecutorDeviceType device_type_;
};

// Keeps track of the multiple window functions in a window query.
class WindowProjectNodeContext {
 public:
  void addWindowFunctionContext(
      std::unique_ptr<WindowFunctionContext> window_function_context,
      const size_t target_index);

  // Marks the window function at the given target index as active. This simplifies the
  // code generation since it's now context sensitive. Each value window function can have
  // its own iteration order, therefore fetching a column at a given position changes
  // depending on which window function is active.
  const WindowFunctionContext* activateWindowFunctionContext(
      const size_t target_index) const;

  // Resets the active window function, which restores the regular (non-window) codegen
  // behavior.
  static void resetWindowFunctionContext();

  // Gets the current active window function.
  static WindowFunctionContext* getActiveWindowFunctionContext();

  // Creates the context for a window function execution unit.
  static WindowProjectNodeContext* create();

  // Retrieves the context for the active window function execution unit.
  static const WindowProjectNodeContext* get();

  // Resets the active context.
  static void reset();

 private:
  // A map from target index to the context associated with the window function at that
  // target index.
  std::unordered_map<size_t, std::unique_ptr<WindowFunctionContext>> window_contexts_;
  // Singleton instance used for an execution unit which is a project with window
  // functions.
  static std::unique_ptr<WindowProjectNodeContext> s_instance_;
  // The active window function. Method comments in this class describe how it's used.
  static WindowFunctionContext* s_active_window_function_;
};
