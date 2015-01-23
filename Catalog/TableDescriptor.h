#ifndef TABLE_DESCRIPTOR_H
#define TABLE_DESCRIPTOR_H

#include <string>
#include <cstdint>
#include "../Shared/sqldefs.h"

/**
 * @type TableDescriptor
 * @brief specifies the content in-memory of a row in the table metadata table
 * 
 */

struct TableDescriptor {
    int32_t tableId; /**< tableId starts at 0 for valid tables. */
    std::string tableName; /**< tableName is the name of the table table -must be unique */
		int32_t nColumns;
		bool isView;
		bool isMaterialized;
		std::string viewSQL;
		std::string fragments;
		std::string partitions;
		ViewStorageOption storageOption; // only relevant to materialized views
		ViewRefreshOption refreshOption; // only relevant to materialized views
		bool checkOption; // only relevant to updateable views.  CHECK OPTION
		bool isReady; // only set at run time when a materialized view is ready to be consumed
};



#endif // TABLE_DESCRIPTOR
