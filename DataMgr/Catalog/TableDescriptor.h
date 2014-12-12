#ifndef TABLE_DESCRIPTOR_H
#define TABLE_DESCRIPTOR_H

#include <string>

/**
 * @type TableDescriptor
 * @brief specifies the content in-memory of a row in the table metadata table
 * 
 * A TableDescriptor type currently includes only the table name and the tableId (zero-based) that it maps to. Other metadata could be added in the future.
 */

struct TableDescriptor {
    std::string tableName; /**< tableName is the name of the table table -must be unique */
    int tableId; /**< tableId starts at 0 for valid tables. */

    TableDescriptor(const std::string &tableName, const int tableId): tableName(tableName), tableId(tableId) {}
    TableDescriptor() {}
};



#endif // TABLE_DESCRIPTOR
