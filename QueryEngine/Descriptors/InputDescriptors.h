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

#ifndef QUERYENGINE_INPUTDESCRIPTORS_H
#define QUERYENGINE_INPUTDESCRIPTORS_H

#include "../Catalog/TableDescriptor.h"
#include "Logger/Logger.h"
#include "Shared/DbObjectKeys.h"
#include "Shared/misc.h"
#include "Shared/toString.h"

#include <memory>

enum class InputSourceType { TABLE, RESULT };

class InputDescriptor {
 public:
  InputDescriptor(int32_t db_id, int32_t table_id, int32_t nest_level)
      : table_key_(db_id, table_id), nest_level_(nest_level) {}

  bool operator==(const InputDescriptor& that) const {
    return table_key_ == that.table_key_ && nest_level_ == that.nest_level_;
  }

  const shared::TableKey& getTableKey() const { return table_key_; }

  int32_t getNestLevel() const { return nest_level_; }

  InputSourceType getSourceType() const;

  size_t hash() const;

  std::string toString() const;

 private:
  shared::TableKey table_key_;
  int32_t nest_level_;
};

inline std::ostream& operator<<(std::ostream& os, InputDescriptor const& id) {
  return os << "InputDescriptor(db_id(" << id.getTableKey().db_id << "), table_id("
            << id.getTableKey().table_id << "),nest_level(" << id.getNestLevel() << "))";
}

class InputColDescriptor final {
 public:
  InputColDescriptor(int32_t col_id, int32_t table_id, int32_t db_id, int32_t nest_level)
      : col_id_(col_id), input_desc_(db_id, table_id, nest_level) {}

  bool operator==(const InputColDescriptor& that) const {
    return col_id_ == that.col_id_ && input_desc_ == that.input_desc_;
  }

  int getColId() const { return col_id_; }

  const InputDescriptor& getScanDesc() const { return input_desc_; }

  shared::TableKey getTableKey() const {
    return shared::TableKey{input_desc_.getTableKey()};
  }

  shared::ColumnKey getColumnKey() const {
    return shared::ColumnKey{getTableKey(), col_id_};
  }

  size_t hash() const {
    return input_desc_.hash() ^ (static_cast<size_t>(col_id_) << 16);
  }

  std::string toString() const {
    return ::typeName(this) + "(col_id=" + std::to_string(col_id_) +
           ", input_desc=" + ::toString(input_desc_) + ")";
  }

 private:
  const int col_id_;
  const InputDescriptor input_desc_;
};

inline std::ostream& operator<<(std::ostream& os, InputColDescriptor const& icd) {
  return os << "InputColDescriptor(col_id(" << icd.getColId() << ")," << icd.getScanDesc()
            << ')';
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
    return input_col_desc.hash();
  }
};

// Used by hash<std::shared_ptr<const InputColDescriptor>>.
template <>
struct hash<const InputColDescriptor*> {
  size_t operator()(const InputColDescriptor* input_col_desc) const {
    CHECK(input_col_desc);
    return input_col_desc->hash();
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
