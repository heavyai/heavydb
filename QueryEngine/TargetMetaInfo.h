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

#ifndef QUERYENGINE_TARGETMETAINFO_H
#define QUERYENGINE_TARGETMETAINFO_H

#include <string>

#include "IR/Type.h"
#include "Shared/sqltypes.h"

/*
 * @type TargetMetaInfo
 * @brief Encapsulates the name and the type of a relational projection.
 */
class TargetMetaInfo {
 public:
  TargetMetaInfo(const std::string& resname, const hdk::ir::Type* type)
      : resname_(resname), type_(type), ti_(type_->toTypeInfo()) {}
  const std::string& get_resname() const { return resname_; }
  const hdk::ir::Type* type() const { return type_; }
  const SQLTypeInfo& get_type_info() const { return ti_; }

  std::string toString() const {
    return "TargetMetaInfo(" + resname_ + ", " + ti_.to_string() + ") ";
  }

 private:
  std::string resname_;
  const hdk::ir::Type* type_;
  const hdk::ir::Type* physical_type_;
  SQLTypeInfo ti_;
  SQLTypeInfo physical_ti_;
};

inline std::ostream& operator<<(std::ostream& os, TargetMetaInfo const& tmi) {
  return os << "TargetMetaInfo(resname_(" << tmi.get_resname() << ") ti_("
            << tmi.get_type_info().to_string() << "))";
}

#endif  // QUERYENGINE_TARGETMETAINFO_H
