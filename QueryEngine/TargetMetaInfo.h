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
  TargetMetaInfo(const std::string& resname, const SQLTypeInfo& ti) : resname_(resname), ti_(ti) {}
  const std::string& get_resname() const { return resname_; }
  const SQLTypeInfo& get_type_info() const { return ti_; }

 private:
  std::string resname_;
  SQLTypeInfo ti_;
};

#endif  // QUERYENGINE_TARGETMETAINFO_H
