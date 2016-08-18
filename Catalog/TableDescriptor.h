#ifndef TABLE_DESCRIPTOR_H
#define TABLE_DESCRIPTOR_H

#include <string>
#include <cstdint>
#include "../Shared/sqldefs.h"
#include "../Fragmenter/AbstractFragmenter.h"

/**
 * @type TableDescriptor
 * @brief specifies the content in-memory of a row in the table metadata table
 *
 */

struct TableDescriptor {
  int32_t tableId;       /**< tableId starts at 0 for valid tables. */
  std::string tableName; /**< tableName is the name of the table table -must be unique */
  int32_t nColumns;
  bool isView;
  bool isMaterialized;
  std::string viewSQL;
  std::string fragments;                          // placeholder for fragmentation information
  Fragmenter_Namespace::FragmenterType fragType;  // fragmentation type. Only INSERT_ORDER is supported now.
  int32_t maxFragRows;                            // max number of rows per fragment
  int64_t maxChunkSize;                           // max number of rows per fragment
  int32_t fragPageSize;                           // page size
  int64_t maxRows;                                // max number of rows in the table
  std::string partitions;                         // placeholder for distributed partition scheme
  StorageOption storageOption;                    // only relevant to materialized views
  ViewRefreshOption refreshOption;                // only relevant to materialized views
  bool checkOption;                               // only relevant to updateable views.  CHECK OPTION
  bool isReady;  // only set at run time when a materialized view is ready to be consumed.  already set to true for
                 // tables.
  Fragmenter_Namespace::AbstractFragmenter*
      fragmenter;  // point to fragmenter object for the table.  it's instantiated upon first use.
};

#endif  // TABLE_DESCRIPTOR
