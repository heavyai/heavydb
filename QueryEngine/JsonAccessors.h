/*
 * @file    JsonAccessors.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Checked json field retrieval.
 *
 */

#ifndef QUERYENGINE_JSONACCESSORS_H
#define QUERYENGINE_JSONACCESSORS_H

#include <glog/logging.h>
#include <rapidjson/document.h>

inline const rapidjson::Value& field(const rapidjson::Value& obj, const char field[]) noexcept {
  CHECK(obj.IsObject());
  const auto field_it = obj.FindMember(field);
  CHECK(field_it != obj.MemberEnd());
  return field_it->value;
}

inline const int64_t json_i64(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsInt64());
  return obj.GetInt64();
}

inline const std::string json_str(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsString());
  return obj.GetString();
}

inline const bool json_bool(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsBool());
  return obj.GetBool();
}

inline const double json_double(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsDouble());
  return obj.GetDouble();
}

#endif  // QUERYENGINE_JSONACCESSORS_H
