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
      : col_id_(col_id),
        input_desc_(table_id, input_desc),
        is_indirect_(false),
        iter_col_id_(-1),
        iter_input_desc_(table_id, -1),
        ref_col_id_(-1),
        ref_input_desc_(table_id, -1) {}

  bool operator==(const InputColDescriptor& that) const {
    return col_id_ == that.col_id_ && input_desc_ == that.input_desc_;
  }

  int getColId() const { return col_id_; }

  const InputDescriptor& getScanDesc() const { return input_desc_; }

  // TODO(miyu) : put the following methods in a subclass
  InputColDescriptor(const int col_id,
                     const int table_id,
                     const int input_desc,
                     const int iter_col_id,
                     const int iter_table_id,
                     const int iter_input_desc,
                     const int ref_col_id,
                     const int ref_table_id,
                     const int ref_input_desc)
      : col_id_(col_id),
        input_desc_(table_id, input_desc),
        is_indirect_(true),
        iter_col_id_(iter_col_id),
        iter_input_desc_(iter_table_id, iter_input_desc),
        ref_col_id_(ref_col_id),
        ref_input_desc_(ref_table_id, ref_input_desc) {}

  bool isIndirect() const { return is_indirect_; }

  int getIterIndex() const {
    CHECK(is_indirect_);
    return iter_col_id_;
  }

  const InputDescriptor& getIterDesc() const {
    CHECK(is_indirect_);
    return iter_input_desc_;
  }

  int getRefColIndex() const {
    CHECK(is_indirect_);
    return ref_col_id_;
  }

  const InputDescriptor& getIndirectDesc() const {
    CHECK(is_indirect_);
    return ref_input_desc_;
  }

 private:
  const int col_id_;
  const InputDescriptor input_desc_;

  // TODO(miyu) : put the following fields in a subclass
  const bool is_indirect_;
  const int iter_col_id_;
  const InputDescriptor iter_input_desc_;

  const int ref_col_id_;
  const InputDescriptor ref_input_desc_;
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
