#include "ColumnInfo.h"

std::string ColumnRef::toString() const {
  return ::typeName(this) + "(db_id=" + std::to_string(db_id) +
         ", table_id=" + std::to_string(table_id) +
         ", column_id=" + std::to_string(column_id);
}

std::string ColumnInfo::toString() const {
  return name + "(db_id=" + std::to_string(db_id) +
         ", table_id=" + std::to_string(table_id) +
         ", column_id=" + std::to_string(column_id) + " type=" + type.toString() +
         (is_rowid ? " [rowid])" : "") + ")";
}