#ifndef TABLE_DESCRIPTOR_H
#define TABLE_DESCRIPTOR_H

#include <string>
#include <cstdint>

/**
 * @type TableDescriptor
 * @brief specifies the content in-memory of a row in the table metadata table
 * 
 */

//namespace Catalog_Namespace {

struct TableDescriptor {
    int32_t tableId; /**< tableId starts at 0 for valid tables. */
    std::string tableName; /**< tableName is the name of the table table -must be unique */
		int32_t nColumns;
		bool isView;
		bool isGPU;
		std::string viewSQL;
		std::string fragments;
		std::string partitions;
};

//}



#endif // TABLE_DESCRIPTOR
