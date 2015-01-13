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
};

#endif // COLUMN_DESCRIPTOR
