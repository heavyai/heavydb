/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Utils/AnyDataType.h"

#include "GfxDriver/Colors/ColorUnion.h"
#include "QueryRenderer/Marks/Enums.h"

namespace QueryRenderer {

void AnyDataType::set(const QueryDataType type, const std::any& val) {
  type_ = type;
  value_ = val;
  size_ = -1;
}

void AnyDataType::convertToType(const QueryDataType new_type) {
  if (new_type == type_) {
    return;
  }

  RUNTIME_EX_ASSERT(isArithmeticQueryDataType(new_type),
                    "Cannot convert from data type: " + to_string(type_) +
                        " to data type: " + to_string(type_) +
                        ". Can only convert between numeric types.");

  switch (new_type) {
    case QueryDataType::UINT:
      convertToType<unsigned int>();
      break;
    case QueryDataType::INT:
      convertToType<int>();
      break;
    case QueryDataType::FLOAT:
      convertToType<float>();
      break;
    case QueryDataType::DOUBLE:
      convertToType<double>();
      break;
    case QueryDataType::UINT64:
      convertToType<uint64_t>();
      break;
    case QueryDataType::INT64:
      convertToType<int64_t>();
      break;
    default:
      CHECK(false) << "Should not have reached this point" << std::endl;
      break;
  }
}

std::string AnyDataType::getStringVal() const {
  RUNTIME_EX_ASSERT(size_ == -1,
                    "Can only call getStringVal() on a singular value AnyDataType "
                    "string. This particular type is a vector of size " +
                        std::to_string(size_) + " of type " + to_string(type_) + ".");

  // TODO(croot): support converting any type to a string
  RUNTIME_EX_ASSERT(
      type_ == QueryDataType::STRING,
      "Cannot extract a string from the AnyDataType object. It is marked as a " +
          to_string(type_) + ".");
  if (!value_.has_value()) {
    return "";
  }

  try {
    return std::any_cast<std::string>(value_);
  } catch (std::bad_any_cast& err) {
    THROW_RUNTIME_EX(
        "Cannot get a string value of the AnyDataType. Its internals have gotten out of "
        "sync. It's labeled as a " +
        to_string(type_) + " but its value is not.");
  }
}

bool AnyDataType::operator==(const AnyDataType& other) const {
  if (type_ != other.type_ || value_.type() != other.value_.type() ||
      size_ != other.size_) {
    return false;
  } else if (!value_.has_value()) {
    // both are empty, so they're equal
    return true;
  }

  try {
    switch (type_) {
      case QueryDataType::INT: {
        if (size_ == -1) {
          return std::any_cast<int>(value_) == std::any_cast<int>(other.value_);
        } else {
          return std::any_cast<const std::vector<int>&>(value_) ==
                 std::any_cast<const std::vector<int>&>(other.value_);
        }
      }
      case QueryDataType::UINT: {
        if (size_ == -1) {
          return std::any_cast<unsigned int>(value_) ==
                 std::any_cast<unsigned int>(other.value_);
        } else {
          return std::any_cast<const std::vector<unsigned int>&>(value_) ==
                 std::any_cast<const std::vector<unsigned int>&>(other.value_);
        }
      }
      case QueryDataType::FLOAT: {
        if (size_ == -1) {
          return std::any_cast<float>(value_) == std::any_cast<float>(other.value_);
        } else {
          return std::any_cast<const std::vector<float>&>(value_) ==
                 std::any_cast<const std::vector<float>&>(other.value_);
        }
      }
      case QueryDataType::INT64: {
        if (size_ == -1) {
          return std::any_cast<int64_t>(value_) == std::any_cast<int64_t>(other.value_);
        } else {
          return std::any_cast<const std::vector<int64_t>&>(value_) ==
                 std::any_cast<const std::vector<int64_t>&>(other.value_);
        }
      }
      case QueryDataType::UINT64: {
        if (size_ == -1) {
          return std::any_cast<uint64_t>(value_) == std::any_cast<uint64_t>(other.value_);
        } else {
          return std::any_cast<const std::vector<uint64_t>&>(value_) ==
                 std::any_cast<const std::vector<uint64_t>&>(other.value_);
        }
      }
      case QueryDataType::DOUBLE: {
        if (size_ == -1) {
          return std::any_cast<double>(value_) == std::any_cast<double>(other.value_);
        } else {
          return std::any_cast<const std::vector<double>&>(value_) ==
                 std::any_cast<const std::vector<double>&>(other.value_);
        }
      }
      case QueryDataType::LINE_JOIN_ENUM:
      case QueryDataType::SYMBOL_SHAPE_ENUM:
      case QueryDataType::ANGLE_UNIT_ENUM: {
        if (size_ == -1) {
          return std::any_cast<int>(value_) == std::any_cast<int>(other.value_);
        } else {
          return std::any_cast<const std::vector<int>&>(value_) ==
                 std::any_cast<const std::vector<int>&>(other.value_);
        }
      }
      case QueryDataType::COLOR: {
        if (size_ == -1) {
          return getColorRef<gfx::ColorUnion>() == other.getColorRef<gfx::ColorUnion>();
        } else {
          return std::any_cast<const std::vector<gfx::ColorUnion>&>(value_) ==
                 std::any_cast<const std::vector<gfx::ColorUnion>&>(other.value_);
        }
      }
      case QueryDataType::STRING: {
        if (size_ == -1) {
          return getStringVal() == other.getStringVal();
        } else {
          return std::any_cast<const std::vector<std::string>&>(value_) ==
                 std::any_cast<const std::vector<std::string>&>(other.value_);
        }
      }
      case QueryDataType::BOOL: {
        if (size_ == -1) {
          return std::any_cast<bool>(value_) == std::any_cast<bool>(other.value_);
        } else {
          return std::any_cast<const std::vector<bool>&>(value_) ==
                 std::any_cast<const std::vector<bool>&>(other.value_);
        }
      }
      case QueryDataType::POLYGON_DOUBLE:
        CHECK(false);
      case QueryDataType::LINE_DOUBLE:
        CHECK(false);
    }
  } catch (std::bad_any_cast& err) {
    THROW_RUNTIME_EX(
        "Cannot compare AnyDataTypes. The internals of one of the objects have gotten "
        "out of sync. The object left "
        "of the operand is labeled as a " +
        to_string(type_) + " and its value is a " + value_.type().name() +
        ". The object to the right of the operand is labeled as a " +
        to_string(other.type_) + " and its value is a " + other.value_.type().name() +
        ".");
  }

  return false;
}

namespace {
template <typename T>
T addAny(const std::any& val1, const std::any& val2) {
  const auto v1 = std::any_cast<T>(val1);
  const auto v2 = std::any_cast<T>(val2);
  if (isNullValue(v2)) {
    // NOTE: this will also handle the case where v1 & v2 are null,
    // which just returns NULL
    return v1;
  } else if (isNullValue(v1)) {
    return v2;
  } else {
    return v1 + v2;
  }
}

template <typename T>
bool minCompareAny(const std::any& lhs, const std::any& rhs) {
  // used as the compare function object for std::min calls
  // Valid values will therefore be less than nulls.
  const auto v1 = std::any_cast<T>(lhs);
  const auto v2 = std::any_cast<T>(rhs);
  if (isNullValue(v2)) {
    return true;
  } else if (isNullValue(v1)) {
    return false;
  }
  return v1 < v2;
}

template <typename T>
bool maxCompareAny(const std::any& lhs, const std::any& rhs) {
  // used as the compare function object for std::max calls.
  // This is still a less-than operator, but it needs to handle nulls differently than the
  // minCompareAny. NULLs in std::max are considered less than valid values in this
  // context.
  const auto v1 = std::any_cast<T>(lhs);
  const auto v2 = std::any_cast<T>(rhs);
  if (isNullValue(v2)) {
    return false;
  } else if (isNullValue(v1)) {
    return true;
  }
  return v1 < v2;
}
}  // namespace

AnyDataType& AnyDataType::operator+=(const QueryRenderer::AnyDataType& other) {
  RUNTIME_EX_ASSERT(size_ == -1 && other.size_ == -1,
                    "Cannot perform operator += on AnyDataType array objects. The object "
                    "left of the operand is of size " +
                        std::to_string(size_) +
                        ". And the object to the right of the operand is of size " +
                        std::to_string(other.size_));

  RUNTIME_EX_ASSERT(type_ == other.type_,
                    "Cannot perform operator += on AnyDataType objects of different "
                    "types. The object left of the "
                    "operand is labeled as a " +
                        to_string(type_) +
                        ". The object to the right of the operand is labeled as a " +
                        to_string(other.type_) + ".");
  if (!value_.has_value()) {
    value_ = other.value_;
  } else if (other.value_.has_value()) {
    try {
      switch (type_) {
        case QueryDataType::INT: {
          value_ = addAny<int>(value_, other.value_);
          break;
        }
        case QueryDataType::UINT: {
          value_ = addAny<unsigned int>(value_, other.value_);
          break;
        }
        case QueryDataType::FLOAT: {
          value_ = addAny<float>(value_, other.value_);
          break;
        }
        case QueryDataType::INT64: {
          value_ = addAny<int64_t>(value_, other.value_);
          break;
        }
        case QueryDataType::UINT64: {
          value_ = addAny<uint64_t>(value_, other.value_);
          break;
        }
        case QueryDataType::DOUBLE: {
          value_ = addAny<double>(value_, other.value_);
          break;
        }
        case QueryDataType::STRING:
          value_ = getStringVal() + other.getStringVal();
          break;
        case QueryDataType::LINE_JOIN_ENUM:
        case QueryDataType::SYMBOL_SHAPE_ENUM:
        case QueryDataType::ANGLE_UNIT_ENUM:
        case QueryDataType::COLOR:
        case QueryDataType::BOOL:
        case QueryDataType::POLYGON_DOUBLE:
        case QueryDataType::LINE_DOUBLE:
          CHECK(false);
      }
    } catch (std::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot perform operator += on AnyDataTypes. The internals of one of the "
          "objects have gotten out of sync. "
          "The object left "
          "of the operand is labeled as a " +
          to_string(type_) + " and its value is a " + value_.type().name() +
          ". The object to the right of the operand is labeled as a " +
          to_string(other.type_) + " and its value is a " + other.value_.type().name() +
          ".");
    }
  }

  return *this;
}

AnyDataType::operator std::string() const {
  std::string rtn = "{type: " + to_string(type_) + " val: ";
  if (!value_.has_value()) {
    rtn += "empty";
  }

  if (size_ == -1) {
    try {
      switch (type_) {
        case QueryDataType::INT:
          rtn += std::to_string(std::any_cast<int>(value_));
          return rtn;
        case QueryDataType::UINT:
          rtn += std::to_string(std::any_cast<unsigned int>(value_));
          return rtn;
        case QueryDataType::FLOAT:
          rtn += std::to_string(std::any_cast<float>(value_));
          return rtn;
        case QueryDataType::INT64:
          rtn += std::to_string(std::any_cast<int64_t>(value_));
          return rtn;
        case QueryDataType::UINT64:
          rtn += std::to_string(std::any_cast<uint64_t>(value_));
          return rtn;
        case QueryDataType::DOUBLE:
          rtn += std::to_string(std::any_cast<double>(value_));
          return rtn;
        case QueryDataType::COLOR:
          rtn += std::string(std::any_cast<gfx::ColorUnion>(value_));
          return rtn;
        case QueryDataType::STRING:
          rtn += std::string(std::any_cast<std::string>(value_));
          return rtn;
        case QueryDataType::BOOL:
          rtn += std::to_string(std::any_cast<bool>(value_));
          return rtn;
        case QueryDataType::LINE_JOIN_ENUM:
          rtn += to_string(static_cast<LineJoinType>(std::any_cast<int>(value_)));
          return rtn;
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          rtn += to_string(static_cast<SymbolShapeType>(std::any_cast<int>(value_)));
          return rtn;
        case QueryDataType::ANGLE_UNIT_ENUM:
          rtn += to_string(static_cast<AngleUnit>(std::any_cast<int>(value_)));
          return rtn;
        case QueryDataType::POLYGON_DOUBLE:
          break;
        case QueryDataType::LINE_DOUBLE:
          break;
      }
    } catch (std::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot convert AnyDataType to a string. Its internals have gotten out of "
          "sync. It's labeled as a " +
          to_string(type_) + " but its value is not.");
    }

    THROW_RUNTIME_EX("Conversion of AnyDataType type " + to_string(type_) +
                     " to a string is not yet supported");
  } else {
    rtn += "vector of size " + std::to_string(size_);
  }
  rtn += "}";

  return rtn;
}

bool AnyDataType::minCompare(const AnyDataType& lhs, const AnyDataType& rhs) {
  RUNTIME_EX_ASSERT(lhs.size_ == -1 && rhs.size_ == -1,
                    "Cannot perform min compare on AnyDataType array objects. The object "
                    "left of the operand is of size " +
                        std::to_string(lhs.size_) +
                        ". And the object to the right of the operand is of size " +
                        std::to_string(rhs.size_));

  RUNTIME_EX_ASSERT(lhs.getType() == rhs.getType(),
                    "Cannot perform min compare on AnyDataType objects of different "
                    "types. The object left of the "
                    "operand is labeled as a " +
                        to_string(lhs.getType()) +
                        ". The object to the right of the operand is labeled as a " +
                        to_string(rhs.getType()) + ".");

  RUNTIME_EX_ASSERT(lhs.value_.has_value() && rhs.value_.has_value(),
                    "Cannot perform min compare on empty AnyDataType objects.");
  try {
    switch (lhs.getType()) {
      case QueryDataType::INT:
        return minCompareAny<int>(lhs.value_, rhs.value_);
      case QueryDataType::UINT:
        return minCompareAny<unsigned int>(lhs.value_, rhs.value_);
      case QueryDataType::FLOAT:
        return minCompareAny<float>(lhs.value_, rhs.value_);
      case QueryDataType::INT64:
        return minCompareAny<int64_t>(lhs.value_, rhs.value_);
      case QueryDataType::UINT64:
        return minCompareAny<uint64_t>(lhs.value_, rhs.value_);
      case QueryDataType::DOUBLE:
        return minCompareAny<double>(lhs.value_, rhs.value_);
      case QueryDataType::STRING:
        return lhs.getStringVal() < rhs.getStringVal();
      case QueryDataType::LINE_JOIN_ENUM:
      case QueryDataType::SYMBOL_SHAPE_ENUM:
      case QueryDataType::ANGLE_UNIT_ENUM:
      case QueryDataType::COLOR:
      case QueryDataType::BOOL:
      case QueryDataType::POLYGON_DOUBLE:
      case QueryDataType::LINE_DOUBLE:
        CHECK(false);
    }
  } catch (std::bad_any_cast& err) {
    THROW_RUNTIME_EX(
        "Cannot perform min compare on AnyDataTypes. The internals of one of the objects "
        "have gotten out of sync. The object left of the operand is labeled as a " +
        to_string(lhs.getType()) + " and its value is a " + lhs.value_.type().name() +
        ". The object to the right of the operand is labeled as a " +
        to_string(rhs.getType()) + " and its value is a " + rhs.value_.type().name() +
        ".");
  }

  return false;
}

bool AnyDataType::maxCompare(const AnyDataType& lhs, const AnyDataType& rhs) {
  RUNTIME_EX_ASSERT(lhs.size_ == -1 && rhs.size_ == -1,
                    "Cannot perform max compare on AnyDataType array objects. The object "
                    "left of the operand is of size " +
                        std::to_string(lhs.size_) +
                        ". And the object to the right of the operand is of size " +
                        std::to_string(rhs.size_));

  RUNTIME_EX_ASSERT(lhs.getType() == rhs.getType(),
                    "Cannot perform max compare on AnyDataType objects of different "
                    "types. The object left of the "
                    "operand is labeled as a " +
                        to_string(lhs.getType()) +
                        ". The object to the right of the operand is labeled as a " +
                        to_string(rhs.getType()) + ".");

  RUNTIME_EX_ASSERT(lhs.value_.has_value() && rhs.value_.has_value(),
                    "Cannot perform max compare on empty AnyDataType objects.");
  try {
    switch (lhs.getType()) {
      case QueryDataType::INT:
        return maxCompareAny<int>(lhs.value_, rhs.value_);
      case QueryDataType::UINT:
        return maxCompareAny<unsigned int>(lhs.value_, rhs.value_);
      case QueryDataType::FLOAT:
        return maxCompareAny<float>(lhs.value_, rhs.value_);
      case QueryDataType::INT64:
        return maxCompareAny<int64_t>(lhs.value_, rhs.value_);
      case QueryDataType::UINT64:
        return maxCompareAny<uint64_t>(lhs.value_, rhs.value_);
      case QueryDataType::DOUBLE:
        return maxCompareAny<double>(lhs.value_, rhs.value_);
      case QueryDataType::STRING:
        return lhs.getStringVal() > rhs.getStringVal();
      case QueryDataType::LINE_JOIN_ENUM:
      case QueryDataType::SYMBOL_SHAPE_ENUM:
      case QueryDataType::ANGLE_UNIT_ENUM:
      case QueryDataType::COLOR:
      case QueryDataType::BOOL:
      case QueryDataType::POLYGON_DOUBLE:
      case QueryDataType::LINE_DOUBLE:
        CHECK(false);
    }
  } catch (std::bad_any_cast& err) {
    THROW_RUNTIME_EX(
        "Cannot perform max compare on AnyDataTypes. The internals of one of the objects "
        "have gotten out of sync. The object left of the operand is labeled as a " +
        to_string(lhs.getType()) + " and its value is a " + lhs.value_.type().name() +
        ". The object to the right of the operand is labeled as a " +
        to_string(rhs.getType()) + " and its value is a " + rhs.value_.type().name() +
        ".");
  }

  return false;
}

}  // namespace QueryRenderer
