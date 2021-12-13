/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef QUERYENGINE_INPUTDESCRIPTORS_H
#define QUERYENGINE_INPUTDESCRIPTORS_H

#include "../Catalog/TableDescriptor.h"
#include "Logger/Logger.h"
#include "Shared/toString.h"

#include <memory>

enum class InputSourceType { TABLE, RESULT };

class InputDescriptor {
 public:
  InputDescriptor(const int table_id, const int nest_level)
      : table_id_(table_id), nest_level_(nest_level) {}

  bool operator==(const InputDescriptor& that) const {
    return table_id_ == that.table_id_ && nest_level_ == that.nest_level_;
  }

  int getTableId() const { return table_id_; }

  int getNestLevel() const { return nest_level_; }

  InputSourceType getSourceType() const {
    return table_id_ > 0 ? InputSourceType::TABLE : InputSourceType::RESULT;
  }

  std::string toString() const {
    return ::typeName(this) + "(table_id=" + std::to_string(table_id_) +
           ", nest_level=" + std::to_string(nest_level_) + ")";
  }

 private:
  int table_id_;
  int nest_level_;
};

inline std::ostream& operator<<(std::ostream& os, InputDescriptor const& id) {
  return os << "InputDescriptor(table_id(" << id.getTableId() << "),nest_level("
            << id.getNestLevel() << "))";
}

namespace std {
template <>
struct hash<InputDescriptor> {
  size_t operator()(const InputDescriptor& input_desc) const {
    return input_desc.getTableId() ^ input_desc.getNestLevel();
  }
};
}  // namespace std

class InputColDescriptor {
 public:
  InputColDescriptor(ColumnInfoPtr col_info, int nest_level)
      : col_info_(std::move(col_info)), nest_level_(nest_level) {}

  bool operator==(const InputColDescriptor& that) const {
    return getColId() == that.getColId() && getTableId() == that.getTableId() &&
           getNestLevel() == that.getNestLevel();
  }

  int getColId() const { return col_info_->column_id; }

  int getTableId() const { return col_info_->table_id; }

  int getNestLevel() const { return nest_level_; }

  InputDescriptor getScanDesc() const { return {col_info_->table_id, nest_level_}; }

  InputSourceType getSourceType() const {
    return col_info_->table_id > 0 ? InputSourceType::TABLE : InputSourceType::RESULT;
  }

  const SQLTypeInfo& getType() const { return col_info_->type; }

  bool isVirtual() const { return col_info_->is_rowid; }

  ColumnInfoPtr getColInfo() const { return col_info_; }

  virtual ~InputColDescriptor() {}

  std::string toString() const {
    return ::typeName(this) + "(table_id=" + std::to_string(getTableId()) +
           ", nest_level=" + std::to_string(getNestLevel()) +
           "col_id=" + std::to_string(getColId()) + (isVirtual() ? "[virt])" : ")");
  }

 private:
  ColumnInfoPtr col_info_;
  int nest_level_;
};

inline std::ostream& operator<<(std::ostream& os, InputColDescriptor const& icd) {
  return os << icd.toString();
}

// For printing RelAlgExecutionUnit::input_col_descs
inline std::ostream& operator<<(std::ostream& os,
                                std::shared_ptr<const InputColDescriptor> const& icd) {
  return os << *icd;
}

namespace std {
template <>
struct hash<InputColDescriptor> {
  size_t operator()(const InputColDescriptor& input_col_desc) const {
    hash<InputDescriptor> input_col_desc_hasher;
    return input_col_desc_hasher(input_col_desc.getScanDesc()) ^
           static_cast<size_t>(input_col_desc.getColId());
  }
};

template <>
struct hash<const InputColDescriptor*> {
  size_t operator()(const InputColDescriptor* input_col_desc) const {
    hash<InputColDescriptor> input_col_desc_hasher;
    CHECK(input_col_desc);
    return input_col_desc_hasher(*input_col_desc);
  }
};

template <>
struct equal_to<shared_ptr<const InputColDescriptor>> {
  bool operator()(shared_ptr<const InputColDescriptor> const& lhs,
                  shared_ptr<const InputColDescriptor> const& rhs) const {
    CHECK(lhs && rhs);
    return *lhs == *rhs;
  }
};
}  // namespace std

#endif  // QUERYENGINE_INPUTDESCRIPTORS_H
