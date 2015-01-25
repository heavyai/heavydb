#ifndef COLUMN_DESCRIPTOR_H
#define COLUMN_DESCRIPTOR_H

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
    EncodingType compression; // compression scheme 
    int comp_param; // compression parameter for certain encoding types
    std::string chunks;

    ColumnDescriptor(const int tableId, const int columnId, const std::string &columnName, const SQLTypeInfo columnType, const EncodingType compression, const int comp_param = 0): tableId(tableId), columnId(columnId), columnName(columnName),columnType(columnType),compression(compression),comp_param(comp_param) {} 
};

#endif // COLUMN_DESCRIPTOR
