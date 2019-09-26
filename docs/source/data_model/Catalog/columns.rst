All Column metadata for each table is stored in ``mapd_columns``. 

Below is the structure of in-memory ``ColumnDescriptor`` object that holds column information in-memory:

.. code-block:: cpp

    struct ColumnDescriptor {
        int tableId;
        int columnId;
        std::string columnName;
        std::string sourceName;
        SQLTypeInfo columnType;
        std::string chunks;
        bool isSystemCol;
        bool isVirtualCol;
        std::string virtualExpr;
        bool isDeletedCol;
        bool isGeoPhyCol{false};

        ColumnDescriptor() : isSystemCol(false), isVirtualCol(false), isDeletedCol(false) {}
        ColumnDescriptor(const int tableId,
                        const int columnId,
                        const std::string& columnName,
                        const SQLTypeInfo columnType)
            : tableId(tableId)
            , columnId(columnId)
            , columnName(columnName)
            , sourceName(columnName)
            , columnType(columnType)
            , isSystemCol(false)
            , isVirtualCol(false)
            , isDeletedCol(false) {}
        ColumnDescriptor(const bool isGeoPhyCol) : ColumnDescriptor() {
            this->isGeoPhyCol = isGeoPhyCol;
        }
    };