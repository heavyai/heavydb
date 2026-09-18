/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Utils/RapidJSONUtils.h"

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "GfxDriver/Colors/ColorUnion.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

namespace {
template <typename FromType,
          typename ToType,
          typename std::enable_if_t<std::is_integral_v<FromType>>* = nullptr>
bool isLosslessIntegralConversion(const AnyDataType& data_type) {
  auto val = data_type.getVal<FromType>();
  return static_cast<FromType>(static_cast<ToType>(val)) == val;
}

template <typename FromType,
          typename ToType,
          typename std::enable_if_t<std::is_floating_point_v<FromType>>* = nullptr>
bool isLosslessFloatingPtConversion(const AnyDataType& data_type) {
  auto val = data_type.getVal<FromType>();
  auto convval = static_cast<FromType>(static_cast<ToType>(val)) == val;
  return convval <= val && convval >= val;
}

std::string to_string(const JSONValueType type) {
  using vt = JSONValueType;
  switch (type) {
    case vt::kBool:
      return "bool";
    case vt::kNumber:
      return "number";
    case vt::kUInt:
      return "unsigned integer";
    case vt::kInt:
      return "integer";
    case vt::kUInt64:
      return "unsigned 64-bit integer";
    case vt::kInt64:
      return "64-bit integer";
    case vt::kDouble:
      return "double";
    case vt::kString:
      return "string";
    case vt::kObject:
      return "object";
    case vt::kArray:
      return "array";
  }
  return std::string();
}

}  // namespace

JSONLocation JSONLocation::operator[](const size_t i) const {
  CHECK(value_) << RapidJSONUtils::getPointerPath(path_);
  return JSONLocation(getRenderSessionRef(), &(*value_)[i], path_.Append(i));
}

JSONLocation JSONLocation::operator[](const std::string& key) const {
  CHECK(value_) << RapidJSONUtils::getPointerPath(path_);
  return JSONLocation(getRenderSessionRef(),
                      &(*value_)[key.c_str()],
                      path_.Append(key.c_str(), key.length()));
}

bool JSONLocation::hasMember(const std::string& key) const {
  if (value_) {
    return value_->HasMember(key.c_str());
  }
  return false;
}

JSONLocation JSONLocation::getMember(const std::string& key) const {
  JSONLocation rtn{
      getRenderSessionRef(), nullptr, path_.Append(key.c_str(), key.length())};
  const auto mitr = value_->FindMember(key.c_str());
  if (mitr != value_->MemberEnd()) {
    rtn.value_ = &mitr->value;
  }
  return rtn;
}

JSONLocation JSONLocation::getMember(const std::string& key,
                                     const JSONValueType expects_type,
                                     const bool is_required) const {
  if (hasMember(key)) {
    auto loc = getMember(key);
    RUNTIME_EX_ASSERT(loc.isType(expects_type),
                      RapidJSONUtils::createJsonParseError(
                          loc,
                          "the \"" + key + "\" member must be of type " +
                              to_string(expects_type) + "."));
    return loc;
  } else {
    RUNTIME_EX_ASSERT(!is_required,
                      RapidJSONUtils::createJsonParseError(
                          *this, "\"" + key + "\" property is required."));
    return JSONLocation();
  }
}

JSONLocation JSONLocation::getArrayMember(const size_t i,
                                          const JSONValueType expects_type) const {
  CHECK(isArray());
  CHECK_LT(i, size());
  auto rtn = (*this)[i];
  RUNTIME_EX_ASSERT(
      rtn.isType(expects_type),
      RapidJSONUtils::createJsonParseError(
          *this, "array member must be of type " + to_string(expects_type) + "."));
  return rtn;
}

std::vector<std::string> JSONLocation::getMemberNames() const {
  std::vector<std::string> rtn(value_ ? value_->MemberCount() : 0);
  if (value_) {
    size_t i = 0;
    for (auto itr = value_->MemberBegin(); itr != value_->MemberEnd(); ++itr, ++i) {
      rtn[i] = itr->name.GetString();
    }
  }
  return rtn;
}

std::string RapidJSONUtils::getObjAsString(const rapidjson::Value& obj) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  obj.Accept(writer);
  return std::string(sb.GetString());
}

JSONRefErrorLogger RapidJSONUtils::createJsonParseError(const JSONLocation& obj_loc,
                                                        std::string&& err_str) {
  return JSONRefErrorLogger(
      obj_loc.getRenderSessionRef(), obj_loc.getPathRef(), std::move(err_str));
}

JSONRefErrorLogger RapidJSONUtils::createJsonParseError(
    const RenderSessionKey& render_session_key,
    const rapidjson::Pointer& obj_path,
    std::string&& err_str) {
  return JSONRefErrorLogger(render_session_key, obj_path, std::move(err_str));
}

QueryDataType RapidJSONUtils::getDataTypeFromJSONObj(const JSONLocation& obj_loc,
                                                     bool support_string) {
  return getAnyDataFromJSONObj(obj_loc, support_string).getType();
}

AnyDataType RapidJSONUtils::getAnyDataFromJSONObj(const JSONLocation& obj_loc,
                                                  bool support_string) {
  AnyDataType rtn;
  const auto& obj = obj_loc.getValueRef();
  rapidjson::Type type = obj.GetType();
  int enum_val;

  switch (type) {
    case rapidjson::kNumberType:
      if (obj.IsInt()) {
        rtn.set(QueryDataType::INT, obj.GetInt());
      } else if (obj.IsUint()) {
        rtn.set(QueryDataType::UINT, obj.GetUint());
      } else if (obj.IsInt64()) {
        rtn.set(QueryDataType::INT64, obj.GetInt64());
      } else if (obj.IsUint64()) {
        rtn.set(QueryDataType::UINT64, obj.GetUint64());
      } else if (obj.IsDouble()) {
        double val = obj.GetDouble();

        // will be a float if it can be losslessly converted to a float
        // NOTE: version 1.1.0 and greater of rapidjson have functions
        // that do this check, IsLosslessFloat()
        double check = static_cast<double>(static_cast<float>(val));
        if (val >= check && val <= check) {
          rtn.set(QueryDataType::FLOAT, static_cast<float>(val));
        } else {
          rtn.set(QueryDataType::DOUBLE, val);
        }
      } else {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
            obj_loc, "RapidJSON number type is not supported."));
      }
      break;
    case rapidjson::kStringType: {
      // TODO(croot): pass valid string types to validate this or provide a validation
      // function as a pointer to validate the strings
      std::string val = obj.GetString();
      if (support_string) {
        rtn.set(QueryDataType::STRING, val);
      } else if (gfx::isColorString(val)) {
        rtn.set(QueryDataType::COLOR, gfx::ColorUnion(val));
      } else if ((enum_val = convertStringToSymbolShapeEnum(val)) >= 0) {
        rtn.set(QueryDataType::SYMBOL_SHAPE_ENUM, enum_val);
      } else if ((enum_val = convertStringToAngleUnitEnum(val)) >= 0) {
        rtn.set(QueryDataType::ANGLE_UNIT_ENUM, enum_val);
      } else {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
            obj_loc, "non-color or non-symbol shape strings are not a supported type."));
      }
      break;
    }
    case rapidjson::kTrueType:
    case rapidjson::kFalseType:
      rtn.set(QueryDataType::BOOL, obj.GetBool());
      break;
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          obj_loc, "type from JSON is unsupported."));
  }

  return rtn;
}

bool RapidJSONUtils::getHigherOrderDataType(QueryDataType& higher_order_output,
                                            const AnyDataType& base_data_type,
                                            const AnyDataType& check_data_type) {
  bool succeeded = false;
  auto set_higher_order_type = [&succeeded,
                                &higher_order_output](auto higher_order_type) {
    higher_order_output = higher_order_type;
    succeeded = true;
  };

  auto base_type = base_data_type.getType();
  auto check_type = check_data_type.getType();

  if (base_type == check_type) {
    set_higher_order_type(base_type);
  }

  switch (base_type) {
    case QueryDataType::FLOAT: {
      if (check_type == QueryDataType::INT) {
        if (isLosslessIntegralConversion<int, float>(check_data_type)) {
          set_higher_order_type(QueryDataType::FLOAT);
        } else {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      } else if (check_type == QueryDataType::UINT) {
        if (isLosslessIntegralConversion<unsigned int, float>(check_data_type)) {
          set_higher_order_type(QueryDataType::FLOAT);
        } else {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      } else if (check_type == QueryDataType::INT64) {
        if (isLosslessIntegralConversion<int64_t, float>(check_data_type)) {
          set_higher_order_type(QueryDataType::FLOAT);
        }
      } else if (check_type == QueryDataType::UINT64) {
        if (isLosslessIntegralConversion<uint64_t, float>(check_data_type)) {
          set_higher_order_type(QueryDataType::FLOAT);
        }
      } else if (check_type == QueryDataType::DOUBLE) {
        set_higher_order_type(QueryDataType::DOUBLE);
      }
      if (succeeded) {
        return succeeded;
      }
      // NOTE: intentionally not breaking the FLOAT case to let
      // INT64/UINT64 convertible to double pass thru
    }
    case QueryDataType::DOUBLE: {
      if (check_type == QueryDataType::UINT || check_type == QueryDataType::INT ||
          check_type == QueryDataType::FLOAT) {
        set_higher_order_type(QueryDataType::DOUBLE);
      } else if (check_type == QueryDataType::INT64) {
        if (isLosslessIntegralConversion<int64_t, double>(check_data_type)) {
          set_higher_order_type(QueryDataType::INT64);
        } else {
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      } else if (check_type == QueryDataType::UINT64) {
        if (isLosslessIntegralConversion<uint64_t, double>(check_data_type)) {
          set_higher_order_type(QueryDataType::UINT64);
        } else {
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      }
    } break;
    case QueryDataType::INT: {
      if (check_type == QueryDataType::UINT) {
        if (isLosslessIntegralConversion<int, unsigned int>(base_data_type)) {
          set_higher_order_type(QueryDataType::UINT);
        } else {
          // the int < 0, so int64 will hold both
          set_higher_order_type(QueryDataType::INT64);
        }
      } else if (check_type == QueryDataType::FLOAT) {
        if (isLosslessIntegralConversion<int, float>(base_data_type)) {
          set_higher_order_type(QueryDataType::FLOAT);
        } else {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      } else if (check_type == QueryDataType::UINT64) {
        if (isLosslessIntegralConversion<int, uint64_t>(base_data_type)) {
          set_higher_order_type(QueryDataType::UINT64);
        } else if (isLosslessIntegralConversion<uint64_t, double>(base_data_type)) {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      } else if (check_type == QueryDataType::INT64 ||
                 check_type == QueryDataType::DOUBLE) {
        set_higher_order_type(check_type);
      }
    } break;
    case QueryDataType::UINT: {
      if (check_type == QueryDataType::INT) {
        if (isLosslessIntegralConversion<int, unsigned int>(check_data_type)) {
          set_higher_order_type(QueryDataType::UINT);
        } else {
          // the int < 0, so int64 will hold both
          set_higher_order_type(QueryDataType::INT64);
        }
      } else if (check_type == QueryDataType::FLOAT) {
        if (isLosslessIntegralConversion<unsigned int, float>(base_data_type)) {
          set_higher_order_type(QueryDataType::FLOAT);
        } else {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      } else if (check_type == QueryDataType::INT64 ||
                 check_type == QueryDataType::UINT64 ||
                 check_type == QueryDataType::DOUBLE) {
        set_higher_order_type(check_type);
      }
    } break;
    case QueryDataType::INT64: {
      if (check_type == QueryDataType::INT || check_type == QueryDataType::UINT) {
        set_higher_order_type(QueryDataType::INT64);
      } else if (check_type == QueryDataType::FLOAT ||
                 check_type == QueryDataType::DOUBLE) {
        if (isLosslessIntegralConversion<int64_t, double>(base_data_type)) {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
        set_higher_order_type(QueryDataType::INT64);
      } else if (check_type == QueryDataType::UINT64) {
        if (isLosslessIntegralConversion<int64_t, uint64_t>(base_data_type)) {
          set_higher_order_type(QueryDataType::UINT64);
        } else if (isLosslessIntegralConversion<int64_t, double>(base_data_type) &&
                   isLosslessIntegralConversion<uint64_t, double>(check_data_type)) {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      }
    } break;
    case QueryDataType::UINT64: {
      if (check_type == QueryDataType::INT) {
        if (isLosslessIntegralConversion<int, uint64_t>(check_data_type)) {
          set_higher_order_type(QueryDataType::UINT64);
        } else if (isLosslessIntegralConversion<uint64_t, double>(base_data_type)) {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
        // let pass thru to throw an error
      } else if (check_type == QueryDataType::UINT) {
        set_higher_order_type(QueryDataType::UINT64);
      } else if (check_type == QueryDataType::FLOAT ||
                 check_type == QueryDataType::DOUBLE) {
        if (isLosslessIntegralConversion<uint64_t, double>(base_data_type)) {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
        if (check_data_type.getVal<double>() >= 0) {
          set_higher_order_type(QueryDataType::UINT64);
        }
        // let pass thru to throw an error
      } else if (check_type == QueryDataType::INT64) {
        if (isLosslessIntegralConversion<int64_t, uint64_t>(check_data_type)) {
          set_higher_order_type(QueryDataType::UINT64);
        } else if (isLosslessIntegralConversion<uint64_t, double>(base_data_type) &&
                   isLosslessIntegralConversion<int64_t, double>(check_data_type)) {
          // double will hold both
          set_higher_order_type(QueryDataType::DOUBLE);
        }
      }
    } break;

    case QueryDataType::COLOR:
    case QueryDataType::STRING:
    case QueryDataType::BOOL:
    case QueryDataType::LINE_JOIN_ENUM:
    case QueryDataType::SYMBOL_SHAPE_ENUM:
    case QueryDataType::ANGLE_UNIT_ENUM:
    case QueryDataType::POLYGON_DOUBLE:
    case QueryDataType::LINE_DOUBLE:
      break;
  }

  return succeeded;
}

bool RapidJSONUtils::getHigherOrderDataType(QueryDataType& higher_order_output,
                                            const QueryDataType base_data_type,
                                            const QueryDataType check_data_type) {
  bool succeeded = false;
  auto set_higher_order_type = [&succeeded,
                                &higher_order_output](auto higher_order_type) {
    higher_order_output = higher_order_type;
    succeeded = true;
  };

  if (base_data_type == check_data_type) {
    set_higher_order_type(base_data_type);
  } else {
    switch (base_data_type)
    case QueryDataType::FLOAT: {
      if (check_data_type == QueryDataType::INT ||
          check_data_type == QueryDataType::UINT) {
        set_higher_order_type(QueryDataType::FLOAT);
      } else {
        set_higher_order_type(QueryDataType::DOUBLE);
      }
      break;
      case QueryDataType::DOUBLE:
        set_higher_order_type(QueryDataType::DOUBLE);
        break;
      case QueryDataType::INT:
        if (check_data_type == QueryDataType::UINT) {
          // if int < 0, so int64 will hold both
          set_higher_order_type(QueryDataType::INT64);
        } else if (check_data_type == QueryDataType::UINT64) {
          set_higher_order_type(QueryDataType::DOUBLE);
        } else if (check_data_type == QueryDataType::FLOAT ||
                   check_data_type == QueryDataType::INT64 ||
                   check_data_type == QueryDataType::DOUBLE) {
          set_higher_order_type(check_data_type);
        }
        break;
      case QueryDataType::UINT:
        if (check_data_type == QueryDataType::INT) {
          // if the int < 0, so int64 will hold both
          set_higher_order_type(QueryDataType::INT64);
        } else if (check_data_type == QueryDataType::FLOAT ||
                   check_data_type == QueryDataType::INT64 ||
                   check_data_type == QueryDataType::UINT64 ||
                   check_data_type == QueryDataType::DOUBLE) {
          set_higher_order_type(check_data_type);
        }
        break;
      case QueryDataType::INT64:
        if (check_data_type == QueryDataType::INT ||
            check_data_type == QueryDataType::UINT) {
          set_higher_order_type(QueryDataType::INT64);
        } else if (check_data_type == QueryDataType::FLOAT ||
                   check_data_type == QueryDataType::DOUBLE) {
          set_higher_order_type(QueryDataType::DOUBLE);
        } else if (check_data_type == QueryDataType::UINT64) {
          set_higher_order_type(QueryDataType::DOUBLE);
        }
        break;
      case QueryDataType::UINT64:
        if (check_data_type == QueryDataType::INT) {
          set_higher_order_type(QueryDataType::DOUBLE);
        } else if (check_data_type == QueryDataType::UINT) {
          set_higher_order_type(QueryDataType::UINT64);
        } else if (check_data_type == QueryDataType::FLOAT ||
                   check_data_type == QueryDataType::DOUBLE) {
          set_higher_order_type(QueryDataType::DOUBLE);
        } else if (check_data_type == QueryDataType::INT64) {
          set_higher_order_type(QueryDataType::DOUBLE);
        }
        break;
      case QueryDataType::COLOR:
      case QueryDataType::STRING:
      case QueryDataType::BOOL:
      case QueryDataType::LINE_JOIN_ENUM:
      case QueryDataType::SYMBOL_SHAPE_ENUM:
      case QueryDataType::ANGLE_UNIT_ENUM:
      case QueryDataType::POLYGON_DOUBLE:
      case QueryDataType::LINE_DOUBLE:
        break;
    }
  }

  return succeeded;
}

rapidjson::Value RapidJSONUtils::valToJSON(
    const std::string& val,
    rapidjson::Document::AllocatorType& allocator) {
  return rapidjson::Value(val.c_str(), val.length(), allocator);
}

rapidjson::Value RapidJSONUtils::valToJSON(
    const AnyDataType& val,
    rapidjson::Document::AllocatorType& allocator) {
  switch (val.getType()) {
    case QueryDataType::INT:
      return valToJSON(val.getVal<int>(), allocator);
    case QueryDataType::UINT:
      return valToJSON(val.getVal<unsigned int>(), allocator);
    case QueryDataType::FLOAT:
      return valToJSON(val.getVal<float>(), allocator);
    case QueryDataType::INT64:
      return valToJSON(val.getVal<int64_t>(), allocator);
    case QueryDataType::UINT64:
      return valToJSON(val.getVal<uint64_t>(), allocator);
    case QueryDataType::DOUBLE:
      return valToJSON(val.getVal<double>(), allocator);
    case QueryDataType::LINE_JOIN_ENUM:
      return valToJSON(
          makeLowerCase(to_string(static_cast<LineJoinType>(val.getVal<int>()))),
          allocator);
    case QueryDataType::SYMBOL_SHAPE_ENUM:
      return valToJSON(
          makeLowerCase(to_string(static_cast<SymbolShapeType>(val.getVal<int>()))),
          allocator);
    case QueryDataType::ANGLE_UNIT_ENUM:
      return valToJSON(
          makeLowerCase(to_string(static_cast<AngleUnit>(val.getVal<int>()))), allocator);
    case QueryDataType::COLOR:
      return valToJSON(val.getColorRef<gfx::ColorUnion>(), allocator);
    case QueryDataType::STRING:
      return valToJSON(val.getStringVal(), allocator);
    case QueryDataType::BOOL:
      return valToJSON(val.getVal<bool>(), allocator);
    case QueryDataType::LINE_DOUBLE:
    case QueryDataType::POLYGON_DOUBLE:
      CHECK(false) << "Unsupported type " << to_string(val.getType());
  }
  return rapidjson::Value();
}
}  // namespace QueryRenderer
