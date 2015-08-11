#ifndef COLUMN_DESCRIPTOR_H
#define COLUMN_DESCRIPTOR_H

#include <cassert>
#include <string>
#include "../Shared/sqltypes.h"

/**
 * @type ColumnDescriptor
 * @brief specifies the content in-memory of a row in the column metadata table
 * 
 */

struct ColumnDescriptor {
    int tableId; 
    int columnId;
    std::string columnName;
    SQLTypeInfo columnType;
    std::string chunks;
    bool isSystemCol;

    ColumnDescriptor(): isSystemCol(false) {}
    ColumnDescriptor(const int tableId, const int columnId, const std::string &columnName, const SQLTypeInfo columnType): tableId(tableId), columnId(columnId), columnName(columnName),columnType(columnType), isSystemCol(false) {
    } 

};

#endif // COLUMN_DESCRIPTOR
