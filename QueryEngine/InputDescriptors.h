#ifndef QUERYENGINE_INPUTDESCRIPTORS_H
#define QUERYENGINE_INPUTDESCRIPTORS_H

#include "../Catalog/TableDescriptor.h"

#include <glog/logging.h>

class ResultRows;

enum class InputSourceType { TABLE, RESULT };

class InputDescriptor {
 public:
  InputDescriptor(const int table_id, const int nest_level) : table_id_(table_id), nest_level_(nest_level) {}

  bool operator==(const InputDescriptor& that) const {
    return table_id_ == that.table_id_ && nest_level_ == that.nest_level_;
  }

  int getTableId() const { return table_id_; }

  int getNestLevel() const { return nest_level_; }

  InputSourceType getSourceType() const { return table_id_ > 0 ? InputSourceType::TABLE : InputSourceType::RESULT; }

 private:
  const int table_id_;
  const int nest_level_;
};

namespace std {
template <>
struct hash<InputDescriptor> {
  size_t operator()(const InputDescriptor& input_desc) const {
    return input_desc.getTableId() ^ input_desc.getNestLevel();
  }
};
}  // std

class InputColDescriptor {
 public:
  InputColDescriptor(const int col_id, const int table_id, const int input_desc)
      : col_id_(col_id), input_desc_(table_id, input_desc) {}

  bool operator==(const InputColDescriptor& that) const {
    return col_id_ == that.col_id_ && input_desc_ == that.input_desc_;
  }

  int getColId() const { return col_id_; }

  const InputDescriptor& getScanDesc() const { return input_desc_; }

 private:
  const int col_id_;
  const InputDescriptor input_desc_;
};

namespace std {
template <>
struct hash<InputColDescriptor> {
  size_t operator()(const InputColDescriptor& input_col_desc) const {
    hash<InputDescriptor> input_col_desc_hasher;
    return input_col_desc_hasher(input_col_desc.getScanDesc()) ^ static_cast<size_t>(input_col_desc.getColId());
  }
};
}  // std

#endif  // QUERYENGINE_INPUTDESCRIPTORS_H
