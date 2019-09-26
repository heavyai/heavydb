Tables are stored in ``mapd_tables``. ``TableDescriptor`` object stores the metadata of each table that can be accessed through `tableDescriptorMap*_` in-memory maps (a member variable of class Catalog).
Below is the structure of ``TableDescriptor``:

.. code-block:: cpp

    struct TableDescriptor {
        int32_t tableId; /**< tableId starts at 0 for valid tables. */
        int32_t shard;
        std::string tableName; /**< tableName is the name of the table table -must be unique */
        int32_t userId;
        int32_t nColumns;
        bool isView;
        std::string viewSQL;
        std::string fragments;  // placeholder for fragmentation information
        Fragmenter_Namespace::FragmenterType
            fragType;            // fragmentation type. Only INSERT_ORDER is supported now.
        int32_t maxFragRows;     // max number of rows per fragment
        int64_t maxChunkSize;    // max number of rows per fragment
        int32_t fragPageSize;    // page size
        int64_t maxRows;         // max number of rows in the table
        std::string partitions;  // distributed partition scheme
        std::string
            keyMetainfo;  // meta-information about shard keys and shared dictionary, as JSON

        Fragmenter_Namespace::AbstractFragmenter*
            fragmenter;  // point to fragmenter object for the table.  it's instantiated upon
                        // first use.
        int32_t
            nShards;  // # of shards, i.e. physical tables for this logical table (default: 0)
        int shardedColumnId;  // Id of the column to be sharded on
        int sortedColumnId;   // Id of the column to be sorted on
        Data_Namespace::MemoryLevel persistenceLevel;
        bool hasDeletedCol;  // Does table has a delete col, Yes (VACUUM = DELAYED)
                            //                              No  (VACUUM = IMMEDIATE)
        // Spi means Sequential Positional Index which is equivalent to the input index in a
        // RexInput node
        std::vector<int> columnIdBySpi_;  // spi = 1,2,3,...

        // write mutex, only to be used inside catalog package
        std::shared_ptr<std::mutex> mutex_;

        TableDescriptor()
            : tableId(-1)
            , shard(-1)
            , nShards(0)
            , shardedColumnId(0)
            , sortedColumnId(0)
            , persistenceLevel(Data_Namespace::MemoryLevel::DISK_LEVEL)
            , hasDeletedCol(true)
            , mutex_(std::make_shared<std::mutex>()) {}
    };

One or several allowed storage options can be specified while creating a table, like ```maxFragRows```, ```nShards```, etc.
