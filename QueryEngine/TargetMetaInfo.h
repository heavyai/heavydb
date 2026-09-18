/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_TARGETMETAINFO_H
#define QUERYENGINE_TARGETMETAINFO_H

#include <string>

#include "../Shared/sqltypes.h"

/*
 * @type TargetMetaInfo
 * @brief Encapsulates the name and the type of a relational projection.
 */
class TargetMetaInfo {
 public:
  TargetMetaInfo(const std::string& resname, const SQLTypeInfo& ti)
      : resname_(resname), ti_(ti), physical_ti_(ti) {}
  TargetMetaInfo(const std::string& resname,
                 const SQLTypeInfo& ti,
                 const SQLTypeInfo& physical_ti)
      : resname_(resname), ti_(ti), physical_ti_(physical_ti) {}
  const std::string& get_resname() const { return resname_; }
  const SQLTypeInfo& get_type_info() const { return ti_; }
  const SQLTypeInfo& get_physical_type_info() const { return physical_ti_; }

  std::string toString() const {
    return "TargetMetaInfo(" + resname_ + ", " + ti_.to_string() + ", " +
           physical_ti_.to_string() + ") ";
  }

 private:
  std::string resname_;
  SQLTypeInfo ti_;
  SQLTypeInfo physical_ti_;
};

inline std::ostream& operator<<(std::ostream& os, TargetMetaInfo const& tmi) {
  return os << "TargetMetaInfo(resname_(" << tmi.get_resname() << ") ti_("
            << tmi.get_type_info().to_string() << ") physical_ti_("
            << tmi.get_physical_type_info().to_string() << "))";
}

#endif  // QUERYENGINE_TARGETMETAINFO_H
