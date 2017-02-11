#ifndef COLUMNAR_RESULTS_H
#define COLUMNAR_RESULTS_H
#include "ResultRows.h"
#include "IteratorTable.h"
#include "SqlTypesLayout.h"

#include "../Shared/checked_alloc.h"

namespace {
inline int64_t fixed_encoding_nullable_val(const int64_t val, const SQLTypeInfo& type_info) {
  if (type_info.get_compression() != kENCODING_NONE) {
    CHECK(type_info.get_compression() == kENCODING_FIXED || type_info.get_compression() == kENCODING_DICT);
    auto logical_ti = get_logical_type_info(type_info);
    if (val == inline_int_null_val(logical_ti)) {
      return inline_fixed_encoding_null_val(type_info);
    }
  }
  return val;
}
}  // namespace

class ColumnarResults {
 public:
  ColumnarResults(const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                  const ResultRows& rows,
                  const size_t num_columns,
                  const std::vector<SQLTypeInfo>& target_types)
      : column_buffers_(num_columns), num_rows_(rows.rowCount()), target_types_(target_types) {
    column_buffers_.resize(num_columns);
    for (size_t i = 0; i < num_columns; ++i) {
      CHECK(!target_types[i].is_array());
      CHECK(!target_types[i].is_string() ||
            (target_types[i].get_compression() == kENCODING_DICT && target_types[i].get_logical_size() == 4));
      column_buffers_[i] = reinterpret_cast<const int8_t*>(checked_malloc(num_rows_ * target_types[i].get_size()));
      row_set_mem_owner->addColBuffer(column_buffers_[i]);
    }
    size_t row_idx{0};
    while (true) {
      const auto crt_row = rows.getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      for (size_t i = 0; i < num_columns; ++i) {
        const auto col_val = crt_row[i];
        const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
        CHECK(scalar_col_val);
        auto i64_p = boost::get<int64_t>(scalar_col_val);
        const auto& type_info = target_types[i];
        if (i64_p) {
          const auto val = fixed_encoding_nullable_val(*i64_p, type_info);
          switch (target_types[i].get_size()) {
            case 1:
              ((int8_t*)column_buffers_[i])[row_idx] = static_cast<int8_t>(val);
              break;
            case 2:
              ((int16_t*)column_buffers_[i])[row_idx] = static_cast<int16_t>(val);
              break;
            case 4:
              ((int32_t*)column_buffers_[i])[row_idx] = static_cast<int32_t>(val);
              break;
            case 8:
              ((int64_t*)column_buffers_[i])[row_idx] = val;
              break;
            default:
              CHECK(false);
          }
        } else {
          CHECK(target_types[i].is_fp());
          switch (target_types[i].get_type()) {
            case kFLOAT: {
              auto float_p = boost::get<float>(scalar_col_val);
              ((float*)column_buffers_[i])[row_idx] = static_cast<float>(*float_p);
              break;
            }
            case kDOUBLE: {
              auto double_p = boost::get<double>(scalar_col_val);
              ((double*)column_buffers_[i])[row_idx] = static_cast<double>(*double_p);
              break;
            }
            default:
              CHECK(false);
          }
        }
      }
      ++row_idx;
    }
    rows.moveToBegin();
  }

  ColumnarResults(const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                  const IteratorTable& table,
                  const int frag_id,
                  const std::vector<SQLTypeInfo>& target_types)
      : num_rows_([&]() {
          auto fragment = table.getFragAt(frag_id);
          CHECK(!fragment.row_count || fragment.data);
          return fragment.row_count;
        }()),
        target_types_(target_types) {
    auto fragment = table.getFragAt(frag_id);
    const auto col_count = table.colCount();
    column_buffers_.resize(col_count);
    if (!num_rows_) {
      return;
    }
    for (size_t i = 0, col_base_off = 0; i < col_count; ++i, col_base_off += num_rows_) {
      CHECK(target_types[i].get_type() == kBIGINT);
      const auto buf_size = num_rows_ * (get_bit_width(target_types[i]) / 8);
      // TODO(miyu): copy offset ptr into frag buffer of 'table' instead of alloc'ing new buffer
      //             if it's proved to survive 'this' b/c it's already columnar.
      column_buffers_[i] = reinterpret_cast<const int8_t*>(checked_malloc(buf_size));
      memcpy(((void*)column_buffers_[i]), &fragment.data[col_base_off], buf_size);
      row_set_mem_owner->addColBuffer(column_buffers_[i]);
    }
  }

  ColumnarResults(const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                  const int8_t* one_col_buffer,
                  const size_t num_rows,
                  const SQLTypeInfo& target_type)
      : column_buffers_(1), num_rows_(num_rows), target_types_{target_type} {
    const auto buf_size = num_rows * get_bit_width(target_type) / 8;
    column_buffers_[0] = reinterpret_cast<const int8_t*>(checked_malloc(buf_size));
    memcpy(((void*)column_buffers_[0]), one_col_buffer, buf_size);
    row_set_mem_owner->addColBuffer(column_buffers_[0]);
  }

#ifdef ENABLE_MULFRAG_JOIN
  static std::unique_ptr<ColumnarResults> createIndexedResults(
      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const std::vector<const ColumnarResults*>& val_frags,
      const std::vector<uint64_t>& frag_offsets,
      const ColumnarResults& indices,
      const int which) {
    const auto idx_buf = reinterpret_cast<const int64_t*>(indices.column_buffers_[which]);
    const auto row_count = indices.num_rows_;
    CHECK_EQ(frag_offsets.size(), val_frags.size());
    CHECK_GT(val_frags.size(), size_t(0));
    CHECK(val_frags[0] != nullptr);
    const auto col_count = val_frags[0]->column_buffers_.size();
    std::unique_ptr<ColumnarResults> filtered_vals(new ColumnarResults(row_count, val_frags[0]->target_types_));
    CHECK(filtered_vals->column_buffers_.empty());
    const auto consist_frag_size = get_consistent_frag_size(frag_offsets);
    for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
      const auto byte_width = get_bit_width(val_frags[0]->getColumnType(col_idx)) / 8;
      auto write_ptr = reinterpret_cast<int8_t*>(checked_malloc(byte_width * row_count));
      filtered_vals->column_buffers_.push_back(write_ptr);
      row_set_mem_owner->addColBuffer(write_ptr);

      for (size_t row_idx = 0; row_idx < row_count; ++row_idx, write_ptr += byte_width) {
        int64_t frag_id = 0;
        int64_t local_idx = idx_buf[row_idx];
        if (val_frags.size() > size_t(1)) {
          if (consist_frag_size != ssize_t(-1)) {
            frag_id = idx_buf[row_idx] / consist_frag_size;
            local_idx = idx_buf[row_idx] % consist_frag_size;
          } else {
            std::tie(frag_id, local_idx) = get_frag_id_and_local_idx(frag_offsets, idx_buf[row_idx]);
          }
        }
        const int8_t* read_ptr = val_frags[frag_id]->column_buffers_[col_idx] + local_idx * byte_width;
        switch (byte_width) {
          case 8:
            *reinterpret_cast<int64_t*>(write_ptr) = *reinterpret_cast<const int64_t*>(read_ptr);
            break;
          case 4:
            *reinterpret_cast<int32_t*>(write_ptr) = *reinterpret_cast<const int32_t*>(read_ptr);
            break;
          case 2:
            *reinterpret_cast<int16_t*>(write_ptr) = *reinterpret_cast<const int16_t*>(read_ptr);
            break;
          case 1:
            *reinterpret_cast<int8_t*>(write_ptr) = *reinterpret_cast<const int8_t*>(read_ptr);
            break;
          default:
            CHECK(false);
        }
      }
    }
    return filtered_vals;
  }
#endif

  static std::unique_ptr<ColumnarResults> createIndexedResults(
      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const ColumnarResults& values,
      const ColumnarResults& indices,
      const int which) {
    const auto idx_buf = reinterpret_cast<const int64_t*>(indices.column_buffers_[which]);
    const auto row_count = indices.num_rows_;
    const auto col_count = values.column_buffers_.size();
    std::unique_ptr<ColumnarResults> filtered_vals(new ColumnarResults(row_count, values.target_types_));
    CHECK(filtered_vals->column_buffers_.empty());
    for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
      const auto byte_width = get_bit_width(values.getColumnType(col_idx)) / 8;
      auto write_ptr = reinterpret_cast<int8_t*>(checked_malloc(byte_width * row_count));
      filtered_vals->column_buffers_.push_back(write_ptr);
      row_set_mem_owner->addColBuffer(write_ptr);

      for (size_t row_idx = 0; row_idx < row_count; ++row_idx, write_ptr += byte_width) {
        const int8_t* read_ptr = values.column_buffers_[col_idx] + idx_buf[row_idx] * byte_width;
        switch (byte_width) {
          case 8:
            *reinterpret_cast<int64_t*>(write_ptr) = *reinterpret_cast<const int64_t*>(read_ptr);
            break;
          case 4:
            *reinterpret_cast<int32_t*>(write_ptr) = *reinterpret_cast<const int32_t*>(read_ptr);
            break;
          case 2:
            *reinterpret_cast<int16_t*>(write_ptr) = *reinterpret_cast<const int16_t*>(read_ptr);
            break;
          case 1:
            *reinterpret_cast<int8_t*>(write_ptr) = *reinterpret_cast<const int8_t*>(read_ptr);
            break;
          default:
            CHECK(false);
        }
      }
    }
    return filtered_vals;
  }

  static std::unique_ptr<ColumnarResults> createOffsetResults(
      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const ColumnarResults& values,
      const int col_idx,
      const uint64_t offset) {
    const auto row_count = values.num_rows_;
    std::unique_ptr<ColumnarResults> offset_vals(new ColumnarResults(row_count, values.target_types_));
    CHECK(offset_vals->column_buffers_.empty());
    CHECK_EQ(64, get_bit_width(values.getColumnType(col_idx)));
    const size_t buf_size = sizeof(int64_t) * row_count;
    auto write_ptr = reinterpret_cast<int64_t*>(checked_malloc(buf_size));
    offset_vals->column_buffers_.push_back(reinterpret_cast<int8_t*>(write_ptr));
    row_set_mem_owner->addColBuffer(write_ptr);
    auto read_ptr = reinterpret_cast<const int64_t*>(values.column_buffers_[col_idx]);
    memcpy(write_ptr, read_ptr, buf_size);
    std::for_each(write_ptr, write_ptr + row_count, [offset](int64_t& n) { n += offset; });
    return offset_vals;
  }

  const std::vector<const int8_t*>& getColumnBuffers() const { return column_buffers_; }

  const size_t size() const { return num_rows_; }

  const SQLTypeInfo& getColumnType(const int col_id) const {
    CHECK_GE(col_id, 0);
    CHECK_LT(static_cast<size_t>(col_id), target_types_.size());
    return target_types_[col_id];
  }

 private:
  ColumnarResults(const size_t num_rows, const std::vector<SQLTypeInfo>& target_types)
      : num_rows_(num_rows), target_types_(target_types) {}

  std::vector<const int8_t*> column_buffers_;
  const size_t num_rows_;
  const std::vector<SQLTypeInfo> target_types_;
};

#endif  // COLUMNAR_RESULTS_H
