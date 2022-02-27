/*
 * Copyright 2022 OmniSci, Inc.
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

#include "StringOpInfo.h"
#include "Logger/Logger.h"

#include <sstream>

namespace StringOps_Namespace {

std::ostream& operator<<(std::ostream& stream, const StringOpInfo& string_op_info) {
  stream << "StringOp("
         << "operator: " << ::toString(string_op_info.getOpKind()) << ", literals: [";
  bool first_elem = true;
  for (const auto& literal_arg : string_op_info.literal_arg_map_) {
    if (!first_elem) {
      stream << ", ";
    }
    first_elem = false;
    const auto datum_type = literal_arg.second.first;
    const auto& datum = literal_arg.second.second;
    stream << "{slot: " << literal_arg.first /* slot/idx */ << ", type: "
           << ::toString(datum_type) << ", value: ";
    if (string_op_info.isLiteralArgNull(datum_type, literal_arg.second.second)) {
      stream << "NULL";
    } else if (IS_STRING(datum_type)) {
      stream << *datum.stringval;
    } else {
      CHECK(IS_INTEGER(datum_type));
      const SQLTypeInfo ti(datum_type, false);
      stream << extract_int_type_from_datum(datum, ti);
    }
    stream << "}";
  }
  stream << "]";
  return stream;
}

std::ostream& operator<<(std::ostream& stream,
                         const std::vector<StringOpInfo>& string_op_infos) {
  stream << "[";
  bool first_elem = true;
  for (const auto& string_op_info : string_op_infos) {
    if (!first_elem) {
      stream << ", ";
    }
    first_elem = false;
    stream << string_op_info;
  }
  stream << "]";
  return stream;
}

std::string toString(const std::vector<StringOpInfo>& string_op_infos) {
  std::ostringstream oss;
  oss << string_op_infos;
  return oss.str();
}

bool StringOpInfo::intLiteralArgAtIdxExists(const size_t index) const {
  const auto literal_itr = literal_arg_map_.find(index);
  if (literal_itr == literal_arg_map_.end()) {
    return false;
  }
  CHECK(IS_INTEGER(literal_itr->second.first));
  return true;
}

bool StringOpInfo::stringLiteralArgAtIdxExists(const size_t index) const {
  const auto literal_itr = literal_arg_map_.find(index);
  if (literal_itr == literal_arg_map_.end()) {
    return false;
  }
  CHECK(IS_STRING(literal_itr->second.first));
  return true;
}

std::string StringOpInfo::getStringLiteral(const size_t index) const {
  const auto str_literal_datum = literal_arg_map_.find(index);
  CHECK(str_literal_datum != literal_arg_map_.end());
  CHECK(IS_STRING(str_literal_datum->second.first));
  CHECK(!StringOpInfo::isLiteralArgNull(str_literal_datum->second.first,
                                        str_literal_datum->second.second));
  return *str_literal_datum->second.second.stringval;
}

int64_t StringOpInfo::getIntLiteral(const size_t index) const {
  const auto literal_datum = literal_arg_map_.find(index);
  CHECK(literal_datum != literal_arg_map_.end());
  const auto& datum_type = literal_datum->second.first;
  CHECK(IS_INTEGER(datum_type));
  const auto& datum = literal_datum->second.second;
  CHECK(!StringOpInfo::isLiteralArgNull(datum_type, datum));
  const SQLTypeInfo ti(datum_type, false);
  return extract_int_type_from_datum(datum, ti);
}

std::string StringOpInfo::toString() const {
  std::ostringstream oss;
  oss << *this;
  return oss.str();
}

//  oss << "StringOp(" << "operator: " << ::toString(getOpKind())
//  << ", literals: [";
//  bool first_arg = true;
//  for (const auto& literal_arg : literal_arg_map_) {
//    const auto datum_type = literal_arg.second.first;
//    const auto& datum = literal_arg.second.second;
//    if (!first_arg) {
//      oss << ", ";
//    }
//    first_arg = false;
//    oss << "{slot: " << literal_arg.first /* slot/idx */ << ", type: "
//     << ::toString(datum_type) << ", value: ";
//    if (isLiteralArgNull(datum_type, literal_arg.second.second)) {
//      oss << "NULL";
//    } else if (IS_STRING(datum_type)) {
//      oss << *datum.stringval;
//    } else {
//      CHECK(IS_INTEGER(datum_type));
//      const SQLTypeInfo ti(datum_type, false);
//      oss << extract_int_type_from_datum(datum, ti);
//    }
//    oss << "}";
//  }
//  return oss.str();
//}

bool StringOpInfo::isLiteralArgNull(const SQLTypes datum_type, const Datum& datum) {
  if (datum_type == kNULLT) {
    CHECK(datum.bigintval == 0);
    return true;
  }
  if (IS_INTEGER(datum_type)) {
    const SQLTypeInfo ti(datum_type, false);
    return ti.is_null(datum);
  }
  CHECK(IS_STRING(datum_type));
  // Currently null strings are empty strings
  // Todo(todd): is this expressed centrally somewhere else in the codebase?
  return datum.stringval == nullptr ? 1UL : 0UL;
}

size_t StringOpInfo::calcNumNullLiteralArgs(const LiteralArgMap& literal_arg_map) {
  size_t num_null_literals{0UL};
  for (const auto& literal_arg : literal_arg_map) {
    const auto& datum_type = literal_arg.second.first;
    const auto& datum = literal_arg.second.second;
    num_null_literals += StringOpInfo::isLiteralArgNull(datum_type, datum) ? 1UL : 0UL;
  }
  return num_null_literals;
}

}  // namespace StringOps_Namespace