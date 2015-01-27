#ifndef TABLE_DESCRIPTOR_H
#define TABLE_DESCRIPTOR_H

#include <string>
#include <cstdint>
#include "../Shared/sqldefs.h"
#include "../Partitioner/Partitioner.h"

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
		std::string fragments; // placeholder for fragmentation information
		Partitioner_Namespace::PartitionerType fragType; // fragmentation type. Only INSERT_ORDER is supported now.
		int32_t maxFragRows; // max number of rows per fragment
		int32_t fragPageSize; // page size
		std::string partitions; // placeholder for distributed partition scheme
		ViewStorageOption storageOption; // only relevant to materialized views
		ViewRefreshOption refreshOption; // only relevant to materialized views
		bool checkOption; // only relevant to updateable views.  CHECK OPTION
		bool isReady; // only set at run time when a materialized view is ready to be consumed
};



#endif // TABLE_DESCRIPTOR
