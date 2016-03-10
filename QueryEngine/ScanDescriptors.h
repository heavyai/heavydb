#ifndef QUERYENGINE_SCANDESCRIPTORS_H
#define QUERYENGINE_SCANDESCRIPTORS_H

#include "../Catalog/TableDescriptor.h"

#include <glog/logging.h>

class ResultRows;

enum class InputSourceType { TABLE, RESULT };

class ScanDescriptor {
 public:
  ScanDescriptor(const int table_id, const int scan_descx) : table_id_(table_id), scan_descx_(scan_descx) {}

  bool operator==(const ScanDescriptor& that) const {
    return table_id_ == that.table_id_ && scan_descx_ == that.scan_descx_;
  }

  int getTableId() const { return table_id_; }

  int getScanIdx() const { return scan_descx_; }

  InputSourceType getSourceType() const { return table_id_ > 0 ? InputSourceType::TABLE : InputSourceType::RESULT; }

 private:
  const int table_id_;
  const int scan_descx_;
};

namespace std {
template <>
struct hash<ScanDescriptor> {
  size_t operator()(const ScanDescriptor& scan_desc) const { return scan_desc.getTableId() ^ scan_desc.getScanIdx(); }
};
}  // std

class ScanColDescriptor {
 public:
  ScanColDescriptor(const int col_id, const int table_id, const int scan_desc)
      : col_id_(col_id), scan_desc_(table_id, scan_desc) {}

  bool operator==(const ScanColDescriptor& that) const {
    return col_id_ == that.col_id_ && scan_desc_ == that.scan_desc_;
  }

  int getColId() const { return col_id_; }

  const ScanDescriptor& getScanDesc() const { return scan_desc_; }

 private:
  const int col_id_;
  const ScanDescriptor scan_desc_;
};

namespace std {
template <>
struct hash<ScanColDescriptor> {
  size_t operator()(const ScanColDescriptor& scan_col_desc) const {
    hash<ScanDescriptor> scan_col_desc_hasher;
    return scan_col_desc_hasher(scan_col_desc.getScanDesc()) ^ static_cast<size_t>(scan_col_desc.getColId());
  }
};
}  // std

#endif  // QUERYENGINE_SCANDESCRIPTORS_H
