#ifndef QUERYENGINE_SCANDESCRIPTORS_H
#define QUERYENGINE_SCANDESCRIPTORS_H

#include "../Catalog/TableDescriptor.h"

#include <glog/logging.h>

enum class InputSourceType { TABLE, RESULT };

class ResultRows;

class ScanColDescriptor {
 public:
  ScanColDescriptor(const int col_id, const TableDescriptor* td, const int scan_idx)
      : input_source_type_(InputSourceType::TABLE), col_id_(col_id), td_(td), scan_idx_(scan_idx) {}

  ScanColDescriptor(const int col_id, const ResultRows* result_rows, const int scan_idx)
      : input_source_type_(InputSourceType::RESULT), col_id_(col_id), result_rows_(result_rows), scan_idx_(scan_idx) {}

  bool operator==(const ScanColDescriptor& that) const {
    const bool payload_equal = input_source_type_ == InputSourceType::TABLE
                                   ? (!!td_ == !!that.td_ && (!td_ || td_->tableId == that.td_->tableId))
                                   : result_rows_ == that.result_rows_;
    return col_id_ == that.col_id_ && payload_equal && scan_idx_ == that.scan_idx_;
  }

  InputSourceType getSourceType() const { return input_source_type_; }

  int getColId() const { return col_id_; }

  const TableDescriptor* getTableDesc() const {
    CHECK(input_source_type_ == InputSourceType::TABLE);
    return td_;
  }

  const ResultRows* getResultRows() const {
    CHECK(input_source_type_ == InputSourceType::RESULT);
    CHECK(result_rows_);
    return result_rows_;
  }

  int getScanIdx() const { return scan_idx_; }

 private:
  const InputSourceType input_source_type_;
  const int col_id_;
  union {
    const TableDescriptor* td_;
    const ResultRows* result_rows_;
  };
  const int scan_idx_;
};

namespace std {
template <>
struct hash<ScanColDescriptor> {
  size_t operator()(const ScanColDescriptor& scan_col_desc) const {
    const size_t payload = scan_col_desc.getSourceType() == InputSourceType::TABLE
                               ? reinterpret_cast<size_t>(scan_col_desc.getTableDesc())
                               : reinterpret_cast<size_t>(scan_col_desc.getResultRows());
    return static_cast<size_t>(scan_col_desc.getColId()) ^ payload ^ scan_col_desc.getScanIdx();
  }
};
}

class ScanDescriptor {
 public:
  ScanDescriptor(const int table_id, const int scan_idx)
      : input_source_type_(InputSourceType::TABLE), table_id_(table_id), scan_idx_(scan_idx) {}

  ScanDescriptor(const ResultRows* result_rows, const int scan_idx)
      : input_source_type_(InputSourceType::RESULT), result_rows_(result_rows), scan_idx_(scan_idx) {}

  bool operator==(const ScanDescriptor& that) const {
    return (input_source_type_ == InputSourceType::TABLE ? table_id_ == that.table_id_
                                                         : result_rows_ == that.result_rows_) &&
           scan_idx_ == that.scan_idx_;
  }

  InputSourceType getSourceType() const { return input_source_type_; }

  int getTableId() const {
    CHECK(input_source_type_ == InputSourceType::TABLE);
    return table_id_;
  }

  const ResultRows* getResultRows() const {
    CHECK(input_source_type_ == InputSourceType::RESULT);
    CHECK(result_rows_);
    return result_rows_;
  }

  int getScanIdx() const { return scan_idx_; }

 private:
  const InputSourceType input_source_type_;
  union {
    const int table_id_;
    const ResultRows* result_rows_;
  };
  const int scan_idx_;
};

namespace std {
template <>
struct hash<ScanDescriptor> {
  size_t operator()(const ScanDescriptor& scan_id) const {
    const size_t table_id = scan_id.getSourceType() == InputSourceType::TABLE
                                ? scan_id.getTableId()
                                : reinterpret_cast<size_t>(scan_id.getResultRows());
    return table_id ^ scan_id.getScanIdx();
  }
};
}

#endif  // QUERYENGINE_SCANDESCRIPTORS_H
