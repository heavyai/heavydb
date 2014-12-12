#ifndef COLUMN_DESCRIPTOR_H
#define COLUMN_DESCRIPTOR_H

/**
 * @type ColumnDescriptor
 * @brief specifies the content in-memory of a row in the column metadata table
 * 
 * A ColumnDescriptor is uniquely identified by a tableId and columnName (or tableId and columnId).  It also specifies the type of the column and whether nulls are allowed. Other metadata could be added in the future
 */

struct ColumnDescriptor {
    int tableId; /**< tableId and columnName constitute the primary key to access rows in the column table - the pair must be unique> */
    std::string columnName;  /**< tableId and columnName constitute the primary key to access rows in the column table - the pair must be unique */
    int columnId;
    mapd_data_t columnType;
    bool notNull; /**< specifies if the column can be null according to SQL standard */


    /**
     * @brief Constructor for populating all member attributes -
     * should be used by Catalog internally.  
     */

    ColumnDescriptor(const int tableId, const std::string columnName, const int columnId, const mapd_data_t columnType, const bool notNull): tableId(tableId), columnName(columnName), columnId(columnId), columnType(columnType), notNull(notNull) {}

    /**
     * @brief Constructor that does not specify tableId or columnId 
     * and columnId - these will be filled in by Catalog.
     * @param columnName name of column
     * @param columnType data type of column
     * @param notNull boolean expressing that no values in column can be null
     */

    ColumnDescriptor(const std::string columnName, const mapd_data_t columnType, const bool notNull): columnName(columnName), columnType(columnType), notNull(notNull), tableId(-1), columnId(-1) {} /**< constructor for adding columns - assumes that tableId and columnId are unknown at this point */
    
    /**
     * @brief   Constructor that requires only the name of the column.
     * @author  Steven Stewart <steve@map-d.com>
     *
     * This constructor was created so that an "empty" ColumnDescriptor object can be declared, 
     * where only the name of the column is known.
     *
     * One use case for this constructor arises during parsing. An AST node has a ColumnDescriptor
     * object. When a column name is parsed, the only known metadata is the name itself --
     * other metadata about the column will be obtained during a tree walking phase in which
     * such nodes are annotated by calling the appropriate Catalog method.
     *
     * @param columnName    The name of the column.
     */
    ColumnDescriptor(const std::string columnName) : columnName(columnName) {}
    
    /**
     * @brief   Prints a representation of the ColumnDescriptor object to stdout
     */
    void print() {
        printf("ColumnDescriptor: tableId=%d columnId=%d columnName=%s columnType=%d notNull=%d\n", tableId, columnId, columnName.c_str(), columnType, notNull);
    }
};

#endif // COLUMN_DESCRIPTOR
