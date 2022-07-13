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

#include "CompilationOptions.h"
#include "DataMgr/DataMgr.h"
#include "Descriptors/RelAlgExecutionDescriptor.h"
#include "ResultSet.h"
#include "TargetMetaInfo.h"
#include "TargetValue.h"

#include <type_traits>

#include "arrow/api.h"
#include "arrow/ipc/api.h"
#ifdef HAVE_CUDA
#include <arrow/gpu/cuda_api.h>
#endif  // HAVE_CUDA

static_assert(ARROW_VERSION >= 16000, "Apache Arrow v0.16.0 or above is required.");

// TODO(wamsi): ValueArray is not optimal. Remove it and inherrit from base vector class.
using ValueArray = boost::variant<std::vector<bool>,
                                  std::vector<int8_t>,
                                  std::vector<int16_t>,
                                  std::vector<int32_t>,
                                  std::vector<int64_t>,
                                  std::vector<arrow::Decimal128>,
                                  std::vector<float>,
                                  std::vector<double>,
                                  std::vector<std::string>,
                                  std::vector<std::vector<int8_t>>,
                                  std::vector<std::vector<int16_t>>,
                                  std::vector<std::vector<int32_t>>,
                                  std::vector<std::vector<int64_t>>,
                                  std::vector<std::vector<float>>,
                                  std::vector<std::vector<double>>,
                                  std::vector<std::vector<std::string>>>;

template <typename T>
using Vec2 = std::vector<std::vector<T>>;

class ArrowResultSet;

class ArrowResultSetRowIterator {
 public:
  using value_type = std::vector<TargetValue>;
  using difference_type = std::ptrdiff_t;
  using pointer = std::vector<TargetValue>*;
  using reference = std::vector<TargetValue>&;
  using iterator_category = std::input_iterator_tag;

  bool operator==(const ArrowResultSetRowIterator& other) const {
    return result_set_ == other.result_set_ && crt_row_idx_ == other.crt_row_idx_;
  }
  bool operator!=(const ArrowResultSetRowIterator& other) const {
    return !(*this == other);
  }

  inline value_type operator*() const;
  ArrowResultSetRowIterator& operator++(void) {
    crt_row_idx_++;
    return *this;
  }
  ArrowResultSetRowIterator operator++(int) {
    ArrowResultSetRowIterator iter(*this);
    ++(*this);
    return iter;
  }

 private:
  const ArrowResultSet* result_set_;
  size_t crt_row_idx_;

  ArrowResultSetRowIterator(const ArrowResultSet* rs)
      : result_set_(rs), crt_row_idx_(0){};

  friend class ArrowResultSet;
};

enum class ArrowTransport { SHARED_MEMORY = 0, WIRE = 1 };

struct ArrowResult {
  std::vector<char> sm_handle;
  int64_t sm_size;
  std::vector<char> df_handle;
  int64_t df_size;
  std::string serialized_cuda_handle;  // Only for GPU memory deallocation
  std::vector<char> df_buffer;         // Only present when transport is WIRE
};

// Expose Arrow buffers as a subset of the ResultSet interface
// to make it work within the existing execution test framework.
class ArrowResultSet {
 public:
  ArrowResultSet(const std::shared_ptr<ResultSet>& rows,
                 const std::vector<TargetMetaInfo>& targets_meta,
                 const ExecutorDeviceType device_type = ExecutorDeviceType::CPU);

  ArrowResultSet(
      const std::shared_ptr<ResultSet>& rows,
      const std::vector<TargetMetaInfo>& targets_meta,
      const ExecutorDeviceType device_type,
      const size_t min_result_size_for_bulk_dictionary_fetch,
      const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);

  ArrowResultSet(const std::shared_ptr<ResultSet>& rows,
                 const ExecutorDeviceType device_type = ExecutorDeviceType::CPU)
      : ArrowResultSet(rows, {}, device_type) {}

  ArrowResultSetRowIterator rowIterator(size_t from_index,
                                        bool translate_strings,
                                        bool decimal_to_double) const {
    ArrowResultSetRowIterator iter(this);
    for (size_t i = 0; i < from_index; i++) {
      ++iter;
    }

    return iter;
  }

  ArrowResultSetRowIterator rowIterator(bool translate_strings,
                                        bool decimal_to_double) const {
    return rowIterator(0, translate_strings, decimal_to_double);
  }

  std::vector<std::string> getDictionaryStrings(const size_t col_idx) const;

  std::vector<TargetValue> getRowAt(const size_t index) const;

  std::vector<TargetValue> getNextRow(const bool translate_strings,
                                      const bool decimal_to_double) const;

  size_t colCount() const;

  SQLTypeInfo getColType(const size_t col_idx) const;

  bool definitelyHasNoRows() const;

  size_t rowCount() const;
  size_t entryCount() const;

  bool isEmpty() const;

  static void deallocateArrowResultBuffer(
      const ArrowResult& result,
      const ExecutorDeviceType device_type,
      const size_t device_id,
      std::shared_ptr<Data_Namespace::DataMgr>& data_mgr);

 private:
  void resultSetArrowLoopback(
      const ExecutorDeviceType device_type = ExecutorDeviceType::CPU);

  void resultSetArrowLoopback(
      const ExecutorDeviceType device_type,
      const size_t min_result_size_for_bulk_dictionary_fetch,
      const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);

  template <typename Type, typename ArrayType>
  void appendValue(std::vector<TargetValue>& row,
                   const arrow::Array& column,
                   const Type null_val,
                   const size_t idx) const;

  std::shared_ptr<ArrowResult> results_;
  std::shared_ptr<ResultSet> rows_;
  std::vector<TargetMetaInfo> targets_meta_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  arrow::ipc::DictionaryMemo dictionary_memo_;

  // Boxed arrays from the record batch. The result of RecordBatch::column is
  // temporary, so we cache these for better performance
  std::vector<std::shared_ptr<arrow::Array>> columns_;
  mutable size_t crt_row_idx_;
  std::vector<TargetMetaInfo> column_metainfo_;
};

ArrowResultSetRowIterator::value_type ArrowResultSetRowIterator::operator*() const {
  return result_set_->getRowAt(crt_row_idx_);
}

class ExecutionResult;

// The following result_set_arrow_loopback methods are used by our test
// framework (ExecuteTest specifically) to take results from the executor,
// serialize them to Arrow and then deserialize them to an ArrowResultSet,
// which can then be used by the test framework.

std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(const ExecutionResult& results);

std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(
    const ExecutionResult* results,
    const std::shared_ptr<ResultSet>& rows,
    const ExecutorDeviceType device_type = ExecutorDeviceType::CPU);

// This version of result_set_arrow_loopback allows setting the parameters that
// drive the choice between dense and sparse dictionary conversion, used for
// Select.ArrowDictionaries tests in ExecuteTest

std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(
    const ExecutionResult* results,
    const std::shared_ptr<ResultSet>& rows,
    const ExecutorDeviceType device_type,
    const size_t min_result_size_for_bulk_dictionary_fetch,
    const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);

enum class ArrowStringRemapMode {
  ALL_STRINGS_REMAPPED,
  ONLY_TRANSIENT_STRINGS_REMAPPED,
  INVALID
};

class ArrowResultSetConverter {
 public:
  static constexpr size_t default_min_result_size_for_bulk_dictionary_fetch{10000UL};
  static constexpr double
      default_max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch{0.1};

  ArrowResultSetConverter(const std::shared_ptr<ResultSet>& results,
                          const std::shared_ptr<Data_Namespace::DataMgr> data_mgr,
                          const ExecutorDeviceType device_type,
                          const int32_t device_id,
                          const std::vector<std::string>& col_names,
                          const int32_t first_n,
                          const ArrowTransport transport_method)
      : results_(results)
      , data_mgr_(data_mgr)
      , device_type_(device_type)
      , device_id_(device_id)
      , col_names_(col_names)
      , top_n_(first_n)
      , transport_method_(transport_method)
      , min_result_size_for_bulk_dictionary_fetch_(
            ArrowResultSetConverter::default_min_result_size_for_bulk_dictionary_fetch)
      , max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch_(
            ArrowResultSetConverter::
                default_max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {}

  ArrowResultSetConverter(
      const std::shared_ptr<ResultSet>& results,
      const std::shared_ptr<Data_Namespace::DataMgr> data_mgr,
      const ExecutorDeviceType device_type,
      const int32_t device_id,
      const std::vector<std::string>& col_names,
      const int32_t first_n,
      const ArrowTransport transport_method,
      const size_t min_result_size_for_bulk_dictionary_fetch,
      const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch)
      : results_(results)
      , data_mgr_(data_mgr)
      , device_type_(device_type)
      , device_id_(device_id)
      , col_names_(col_names)
      , top_n_(first_n)
      , transport_method_(transport_method)
      , min_result_size_for_bulk_dictionary_fetch_(
            min_result_size_for_bulk_dictionary_fetch)
      , max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch_(
            max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {}

  ArrowResult getArrowResult() const;

  // TODO(adb): Proper namespacing for this set of functionality. For now, make this
  // public and leverage the converter class as namespace
  struct ColumnBuilder {
    using StrId = int32_t;
    using ArrowStrId = int32_t;

    std::shared_ptr<arrow::Field> field;
    std::unique_ptr<arrow::ArrayBuilder> builder;
    std::shared_ptr<arrow::StringArray> string_array;
    SQLTypeInfo col_type;
    SQLTypes physical_type;
    ArrowStringRemapMode string_remap_mode{ArrowStringRemapMode::INVALID};
    std::unordered_map<StrId, ArrowStrId> string_remapping;
  };

  ArrowResultSetConverter(const std::shared_ptr<ResultSet>& results,
                          const std::vector<std::string>& col_names,
                          const int32_t first_n)
      : results_(results)
      , col_names_(col_names)
      , top_n_(first_n)
      , min_result_size_for_bulk_dictionary_fetch_(
            ArrowResultSetConverter::default_min_result_size_for_bulk_dictionary_fetch)
      , max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch_(
            ArrowResultSetConverter::
                default_max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {}

  ArrowResultSetConverter(
      const std::shared_ptr<ResultSet>& results,
      const std::vector<std::string>& col_names,
      const int32_t first_n,
      const size_t min_result_size_for_bulk_dictionary_fetch,
      const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch)
      : results_(results)
      , col_names_(col_names)
      , top_n_(first_n)
      , min_result_size_for_bulk_dictionary_fetch_(
            min_result_size_for_bulk_dictionary_fetch)
      , max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch_(
            max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {}

  std::shared_ptr<arrow::RecordBatch> convertToArrow() const;

 private:
  std::shared_ptr<arrow::RecordBatch> getArrowBatch(
      const std::shared_ptr<arrow::Schema>& schema) const;

  std::shared_ptr<arrow::Field> makeField(const std::string name,
                                          const SQLTypeInfo& target_type) const;

  struct SerializedArrowOutput {
    std::shared_ptr<arrow::Buffer> schema;
    std::shared_ptr<arrow::Buffer> records;
  };
  SerializedArrowOutput getSerializedArrowOutput(
      arrow::ipc::DictionaryFieldMapper* mapper) const;

  void initializeColumnBuilder(ColumnBuilder& column_builder,
                               const SQLTypeInfo& col_type,
                               const size_t result_col_idx,
                               const std::shared_ptr<arrow::Field>& field) const;

  void append(ColumnBuilder& column_builder,
              const ValueArray& values,
              const std::shared_ptr<std::vector<bool>>& is_valid) const;

  inline std::shared_ptr<arrow::Array> finishColumnBuilder(
      ColumnBuilder& column_builder) const;

  std::shared_ptr<ResultSet> results_;
  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_ = nullptr;
  ExecutorDeviceType device_type_ = ExecutorDeviceType::GPU;
  int32_t device_id_ = 0;
  std::vector<std::string> col_names_;
  int32_t top_n_;
  ArrowTransport transport_method_;
  const size_t min_result_size_for_bulk_dictionary_fetch_;
  const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch_;
  friend class ArrowResultSet;
};

template <typename T>
constexpr auto scale_epoch_values() {
  return std::is_same<T, arrow::Date32Builder>::value ||
         std::is_same<T, arrow::Date64Builder>::value;
}
