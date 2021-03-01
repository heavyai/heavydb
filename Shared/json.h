// Copyright (c) 2021 OmniSci, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// OmniSci JSON

// EXAMPLE #1
//
//   #include "Shared/json.h"
//   JSON json;
//   json["item1"] = "abc";
//   json["item2"] = 123;
//   std::cout << json.str() << std::endl;
//   std::cout << static_cast<int>(json["item2"]) << std::endl;
//
// OUTPUT: {"item1":"abc","item2":123}
// OUTPUT: 123

// EXAMPLE #2
//
//   #include "Shared/json.h"
//   std::string text = R"json(
//     {
//       "item1": "abc",
//       "item2": 123
//     }
//   )json";
//   JSON json(text);
//   json["item3"] = false;
//   json["item4"].parse("[0, 1, 2, 3, 4]");
//   std::cout << json.str() << std::endl;
//   std::cout << static_cast<size_t>(json["item4"][2]) << std::endl;
//
// OUTPUT: {"item1":"abc","item2":123,"item3":false,"item4":[0,1,2,3,4]}
// OUTPUT: 2

#pragma once

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

class JSON final {
  std::shared_ptr<rapidjson::Document> doc_;
  rapidjson::Value* vptr_;
  rapidjson::Document::AllocatorType& allo_;
  const std::string name_;  // only used in error messages

 public:
  JSON()
      : doc_(std::make_shared<rapidjson::Document>())
      , vptr_(&*doc_)
      , allo_(doc_->GetAllocator())
      , name_("JSON") {}

  JSON(const JSON& peer) : JSON() { vptr_->CopyFrom(*peer.vptr_, allo_); }

  JSON(JSON&&) = default;

  JSON(const std::string& json) : JSON() { parse(json); }

  JSON(const char* json) : JSON() { parse(json); }

  JSON(const char* json, size_t len) : JSON() { parse(json, len); }

  void parse(const std::string& json) {
    if (doc_->Parse(json).HasParseError()) {
      throw std::runtime_error("failed to parse json");
    }
  }

  void parse(const char* json) {
    if (doc_->Parse(json).HasParseError()) {
      throw std::runtime_error("failed to parse json");
    }
  }

  void parse(const char* json, size_t len) {
    if (doc_->Parse(json, len).HasParseError()) {
      throw std::runtime_error("failed to parse json");
    }
  }

  std::string str() const {
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> wr(buf);
    vptr_->Accept(wr);
    return buf.GetString();
  }

  std::string getType() const { return kTypeNames[vptr_->GetType()]; }

  operator std::string() const {
    if (!vptr_->IsString()) {
      throw std::runtime_error("expected JSON field '" + name_ +
                               "' to be String but got [" + kTypeNames[vptr_->GetType()] +
                               "]");
    }
    return std::string{vptr_->GetString(), vptr_->GetStringLength()};
  }

  operator bool() const {
    if (!vptr_->IsBool()) {
      throw std::runtime_error("expected JSON field '" + name_ +
                               "' to be Boolean but got [" +
                               kTypeNames[vptr_->GetType()] + "]");
    }
    return vptr_->GetBool();
  }

  template <typename T>
  operator T() const {
    static_assert(std::numeric_limits<T>::is_integer &&
                  !std::is_same_v<bool, std::remove_cv_t<T>>);
    if constexpr (std::numeric_limits<T>::is_signed) {
      if constexpr (sizeof(T) < 8) {
        if (!vptr_->IsInt()) {
          throw std::runtime_error("expected JSON field '" + name_ +
                                   "' to be signed integer but got [" +
                                   kTypeNames[vptr_->GetType()] + "]");
        }
        return vptr_->GetInt();
      } else {
        if (!vptr_->IsInt64()) {
          throw std::runtime_error("expected JSON field '" + name_ +
                                   "' to be signed 64-bit integer but got [" +
                                   kTypeNames[vptr_->GetType()] + "]");
        }
        return vptr_->GetInt64();
      }
    } else {
      if constexpr (sizeof(T) < 8) {
        if (!vptr_->IsUint()) {
          throw std::runtime_error("expected JSON field '" + name_ +
                                   "' to be unsigned integer but got [" +
                                   kTypeNames[vptr_->GetType()] + "]");
        }
        return vptr_->GetUint();
      } else {
        if (!vptr_->IsUint64()) {
          throw std::runtime_error("expected JSON field '" + name_ +
                                   "' to be unsigned 64-bit integer but got [" +
                                   kTypeNames[vptr_->GetType()] + "]");
        }
        return vptr_->GetUint64();
      }
    }
  }

  JSON& operator=(const JSON& peer) {
    vptr_->CopyFrom(*peer.vptr_, allo_);
    return *this;
  }

  JSON& operator=(const std::string& item) {
    *vptr_ = rapidjson::Value().SetString(item, allo_);
    return *this;
  }

  JSON& operator=(const char* item) {
    *vptr_ = rapidjson::Value().SetString(item, allo_);
    return *this;
  }

  JSON& operator=(bool item) {
    vptr_->SetBool(item);
    return *this;
  }

  JSON& operator=(int32_t item) {
    vptr_->SetInt(item);
    return *this;
  }

  JSON& operator=(int64_t item) {
    vptr_->SetInt64(item);
    return *this;
  }

  JSON& operator=(uint32_t item) {
    vptr_->SetUint(item);
    return *this;
  }

  JSON& operator=(uint64_t item) {
    vptr_->SetUint64(item);
    return *this;
  }

  JSON operator[](const std::string& name) { return (*this)[name.c_str()]; }
  JSON operator[](const char* name) {
    if (!vptr_->IsObject()) {
      vptr_->SetObject();
    }
    if (!vptr_->HasMember(name)) {
      vptr_->AddMember(
          rapidjson::Value(name, allo_).Move(), rapidjson::Value().Move(), allo_);
      auto f = vptr_->FindMember(name);
      // f necessary because AddMember inexplicably doesn't return the new member
      // https://stackoverflow.com/questions/52113291/which-object-reference-does-genericvalueaddmember-return
      return JSON(doc_, &f->value, allo_, name);
    }
    return JSON(doc_, &(*vptr_)[name], allo_, name);
  }

  JSON operator[](const std::string& name) const { return (*this)[name.c_str()]; }
  JSON operator[](const char* name) const {
    if (!vptr_->IsObject()) {
      throw std::runtime_error("JSON " + getType() + " field '" + name_ +
                               "' can't use operator []");
    }
    if (!vptr_->HasMember(name)) {
      throw std::runtime_error("JSON field '" + std::string(name) + "' not found");
    }
    return JSON(doc_, &(*vptr_)[name], allo_, name);
  }

  template <typename T>
  JSON operator[](T index) {
    return operator[](static_cast<size_t>(index));
  }
  JSON operator[](size_t index) {
    if (!vptr_->IsArray()) {
      vptr_->SetArray();
    }
    if (index >= vptr_->Size()) {
      throw std::runtime_error("JSON array index " + std::to_string(index) +
                               " out of range " + std::to_string(vptr_->Size()));
    }
    return JSON(doc_, &(*vptr_)[index], allo_, std::to_string(index));
  }

  template <typename T>
  JSON operator[](T index) const {
    return operator[](static_cast<size_t>(index));
  }
  JSON operator[](size_t index) const {
    if (!vptr_->IsArray()) {
      throw std::runtime_error("JSON " + getType() + " field '" + name_ +
                               "' can't use operator []");
    }
    if (index >= vptr_->Size()) {
      throw std::runtime_error("JSON array index " + std::to_string(index) +
                               " out of range " + std::to_string(vptr_->Size()));
    }
    return JSON(doc_, &(*vptr_)[index], allo_, std::to_string(index));
  }

 private:
  inline static std::string kTypeNames[] =
      {"Null", "False", "True", "Object", "Array", "String", "Number"};

  JSON(std::shared_ptr<rapidjson::Document> doc,
       rapidjson::Value* vptr,
       rapidjson::Document::AllocatorType& allo,
       const std::string& name)
      : doc_(doc), vptr_(vptr), allo_(allo), name_(name) {}
};  // struct JSON
