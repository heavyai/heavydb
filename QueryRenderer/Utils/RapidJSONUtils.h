/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>
#include <rapidjson/writer.h>

#include "GfxDriver/Colors/Types.h"
#include "GfxDriver/RenderError.h"
#include "QueryRenderer/Interface/RenderSessionKey.h"
#include "QueryRenderer/Marks/Enums.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/JSONRefErrorLogger.h"

namespace QueryRenderer {

enum class JSONValueType {
  kBool,
  kNumber,
  kUInt,
  kInt,
  kUInt64,
  kInt64,
  kDouble,
  kString,
  kObject,
  kArray
};

class JSONLocation {
 public:
  static inline std::string getPointerPath(const rapidjson::Pointer& ptr) {
    rapidjson::StringBuffer sb;
    ptr.Stringify(sb);
    return sb.GetString();
  }

  JSONLocation() : session_{nullptr}, value_{nullptr}, path_() {}
  explicit JSONLocation(const RenderSessionKey& in_session,
                        const rapidjson::Value* const in_value,
                        const rapidjson::Pointer& in_path)
      : session_{&in_session}, value_{in_value}, path_(in_path) {}
  JSONLocation(const JSONLocation& other)
      : session_{other.session_}, value_{other.value_}, path_(other.path_) {}
  JSONLocation(JSONLocation&& other)
      : session_{std::move(other.session_)}
      , value_{std::move(other.value_)}
      , path_{std::move(other.path_)} {}

  JSONLocation& operator=(const JSONLocation& other) {
    session_ = other.session_;
    value_ = other.value_;
    path_ = other.path_;
    return *this;
  }

  JSONLocation& operator=(JSONLocation&& other) {
    session_ = std::move(other.session_);
    value_ = std::move(other.value_);
    path_ = std::move(other.path_);
    return *this;
  }

  inline bool isValid() const { return value_ != nullptr; }
  inline bool isBool() const { return value_ ? value_->IsBool() : false; }
  inline bool isNumber() const { return value_ ? value_->IsNumber() : false; }
  inline bool isUint() const { return value_ ? value_->IsUint() : false; }
  inline bool isInt() const { return value_ ? value_->IsInt() : false; }
  inline bool isUint64() const { return value_ ? value_->IsUint64() : false; }
  inline bool isInt64() const { return value_ ? value_->IsInt64() : false; }
  inline bool isDouble() const { return value_ ? value_->IsDouble() : false; }
  inline bool isString() const { return value_ ? value_->IsString() : false; }
  inline bool isObject() const { return value_ ? value_->IsObject() : false; }
  inline bool isArray() const { return value_ ? value_->IsArray() : false; }

  inline bool isType(const JSONValueType type) const {
    using vt = JSONValueType;
    switch (type) {
      case vt::kBool:
        return value_ ? value_->IsBool() : false;
      case vt::kNumber:
        return value_ ? value_->IsNumber() : false;
      case vt::kUInt:
        return value_ ? value_->IsUint() : false;
      case vt::kInt:
        return value_ ? value_->IsInt() : false;
      case vt::kUInt64:
        return value_ ? value_->IsUint64() : false;
      case vt::kInt64:
        return value_ ? value_->IsInt64() : false;
      case vt::kDouble:
        return value_ ? value_->IsDouble() : false;
      case vt::kString:
        return value_ ? value_->IsString() : false;
      case vt::kObject:
        return value_ ? value_->IsObject() : false;
      case vt::kArray:
        return value_ ? value_->IsArray() : false;
    }
    return false;
  }

  inline bool getBool() const {
    CHECK(value_) << getPointerPath(path_);
    return value_->GetBool();
  }
  inline int32_t getInt() const {
    CHECK(value_) << getPointerPath(path_);
    return value_->GetInt();
  }
  inline uint32_t getUint() const {
    CHECK(value_) << getPointerPath(path_);
    return value_->GetUint();
  }
  inline int64_t getInt64() const {
    CHECK(value_) << getPointerPath(path_);
    return value_->GetInt64();
  }
  inline uint64_t getUint64() const {
    CHECK(value_) << getPointerPath(path_);
    return value_->GetUint64();
  }
  inline double getDouble() const {
    CHECK(value_) << getPointerPath(path_);
    return value_->GetDouble();
  }

  inline std::string getString() const {
    CHECK(value_) << getPointerPath(path_);
    return value_->GetString();
  }

  JSONLocation operator[](const size_t i) const;
  JSONLocation operator[](const std::string& key) const;

  bool hasMember(const std::string& key) const;
  JSONLocation getMember(const std::string& key) const;
  JSONLocation getMember(const std::string& key,
                         const JSONValueType expects_type,
                         const bool is_required) const;
  JSONLocation getArrayMember(const size_t i, const JSONValueType expects_type) const;

  std::vector<std::string> getMemberNames() const;

  size_t size() const { return (value_ ? value_->Size() : 0); }

  inline const RenderSessionKey& getRenderSessionRef() const {
    CHECK(session_);
    return *session_;
  }
  inline const rapidjson::Value& getValueRef() const {
    CHECK(value_) << getPointerPath(path_);
    return *value_;
  }

  inline const rapidjson::Pointer& getPathRef() const { return path_; }

 private:
  const RenderSessionKey* session_;
  const rapidjson::Value* value_;
  rapidjson::Pointer path_;
};

struct RapidJSONUtils {
  inline static std::string getNullStr() { return "NULL"; }
  static inline std::string getPointerPath(const rapidjson::Pointer& ptr) {
    return JSONLocation::getPointerPath(ptr);
  }

  static inline bool isValidPath(const rapidjson::Pointer& ptr) {
    return ptr.GetTokenCount() > 0;
  }

  static std::string getObjAsString(const rapidjson::Value& obj);

  static JSONRefErrorLogger createJsonParseError(const JSONLocation& obj_loc,
                                                 std::string&& errStr);
  static JSONRefErrorLogger createJsonParseError(
      const RenderSessionKey& render_session_key,
      const rapidjson::Pointer& obj_path,
      std::string&& err_str);

  static QueryDataType getDataTypeFromJSONObj(const JSONLocation& obj_loc,
                                              bool support_string = false);
  static AnyDataType getAnyDataFromJSONObj(const JSONLocation& obj_loc,
                                           bool support_string = false);
  static bool getHigherOrderDataType(QueryDataType& higher_order_output,
                                     const AnyDataType& base_data_type,
                                     const AnyDataType& check_data_type);
  static bool getHigherOrderDataType(QueryDataType& higher_order_output,
                                     const QueryDataType base_data_type,
                                     const QueryDataType check_data_type);

  template <typename T>
  static T getNumValFromJSONObj(const JSONLocation& obj_loc) {
    RUNTIME_EX_ASSERT(obj_loc.isNumber() || obj_loc.isBool() || obj_loc.isString(),
                      createJsonParseError(obj_loc,
                                           "getNumValFromJSONObj(): rapidjson object is "
                                           "not a number. Cannot get a number value."));

    T rtn(0);

    // TODO: do min/max checks?
    // How would we do this? implicit conversions apparently
    // take place in >=< operations and therefore
    // min/max checks here wouldn't work. For now, implicitly
    // coverting between possible types, but it could result
    // in undefined/unexpected behavior.
    // One way is to have a template specialization for
    // each basic type. Might be the only way to fix.
    // T max = std::numeric_limits<T>::max();
    // T min = std::numeric_limits<T>::lowest();

    const auto& obj = obj_loc.getValueRef();
    if (obj.IsBool()) {
      auto val = obj.GetBool();
      rtn = static_cast<T>(val);
    } else if (obj.IsInt()) {
      auto val = obj.GetInt();
      rtn = static_cast<T>(val);
    } else if (obj.IsUint()) {
      auto val = obj.GetUint();
      rtn = static_cast<T>(val);
    } else if (obj.IsInt64()) {
      auto val = obj.GetInt64();
      rtn = static_cast<T>(val);
    } else if (obj.IsUint64()) {
      auto val = obj.GetUint64();
      rtn = static_cast<T>(val);
    } else if (obj.IsDouble()) {
      auto val = obj.GetDouble();
      rtn = static_cast<T>(val);
    } else if (obj.IsString()) {
      // TODO(croot): this would only be called in int/uint cases.
      // could make specialized funcs to thin out the other types
      auto val = obj.GetString();
      auto enumVal = convertStringToSymbolShapeEnum(val);
      RUNTIME_EX_ASSERT(
          enumVal >= 0,
          createJsonParseError(obj_loc, "json object is not a valid symbol shape"));
      return static_cast<T>(enumVal);
    }

    return rtn;
  }

  template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
  static rapidjson::Value valToJSON(const T val,
                                    rapidjson::Document::AllocatorType& allocator) {
    if (val == getNullValue<T>()) {
      return valToJSON(getNullStr(), allocator);
    }
    return rapidjson::Value(val);
  }

  static rapidjson::Value valToJSON(const std::string& val,
                                    rapidjson::Document::AllocatorType& allocator);

  template <class T,
            typename std::enable_if<std::integral_constant<
                bool,
                gfx::is_color<T>::value || gfx::is_color_union<T>::value>::value>::type* =
                nullptr>
  static rapidjson::Value valToJSON(const T& val,
                                    rapidjson::Document::AllocatorType& allocator) {
    return valToJSON(std::string(val), allocator);
  }

  static rapidjson::Value valToJSON(const AnyDataType& val,
                                    rapidjson::Document::AllocatorType& allocator);
};

}  // namespace QueryRenderer
