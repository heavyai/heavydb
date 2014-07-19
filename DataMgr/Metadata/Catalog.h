/**
 * @file    Catalog.h
 * @author  Todd Mostak <todd@map-d.com>
 * @brief   This file contains the class specification and related data structures for Catalog.
 *
 * This file contains the Catalog class specification. The Catalog class is responsible for storing metadata
 * about stored objects in the system (currently just relations).  At this point it does not take advantage of the 
 * database storage infrastructure; this likely will change in the future as the buildout continues. Although it persists
 * the metainfo on disk, at database startup it reads everything into in-memory dictionaries for fast access.
 *
 */

#ifndef CATALOG_H
#define CATALOG_H

#include <string>
#include <tuple>
#include <map>
#include <vector>

#include "../../Shared/errors.h"

enum ColumnType {DUMMY_TYPE};

/**
 * @type TableRow
 * @brief specifies the content in-memory of a row in the table metadata table
 * 
 * A TableRow type currently includes only the table name and the tableId (zero-based) that it maps to. Other metadata could be added in the future
 */

struct TableRow {
    std::string tableName; /**< tableId is the primary key to access rows in the table table -must bew unique */
    int tableId; /**< tableId starts at 0 for valid tables. */


    TableRow(const std::string &tableName, const int tableId): tableName(tableName), tableId(tableId) {}
};

/**
 * @type ColumnRow
 * @brief specifies the content in-memory of a row in the column metadata table
 * 
 * A ColumnRow is uniquely identified by a tableId and columnName (or tableId and columnId).  It also specifies the type of the column and whether nulls are allowed. Other metadata could be added in the future
 */
struct ColumnRow {
    int tableId; /**< tableId and columnName constitute the primary key to access rows in the column table - the pair must be unique> */
    std::string columnName;  /**< tableId and columnName constitute the primary key to access rows in the column table - the pair must be unique> */
    int columnId;
    ColumnType columnType;
    bool notNull; /**< specifies if the column can be null according to SQL standard */
    ColumnRow(const int tableId, const std::string columnName, const int columnId, const ColumnType columnType, const bool notNull): tableId(tableId), columnName(columnName), columnId(columnId), columnType(columnType), notNull(notNull) {}

    ColumnRow(const std::string columnName, const ColumnType columnType, const bool notNull): columnName(columnName), columnType(columnType), notNull(notNull), tableId(-1), columnId(-1) {} /**< constructor for adding columns - assumes that tableId and columnId are unknown at this point */
};

/**
 * @type TableRowMap
 * @brief Maps table names to pointers to table row structs 
 */

typedef std::map<std::string, TableRow *> TableRowMap;

/**
 * @type ColumnKey
 * @brief ColumnKey is composed of the integer tableId and the string name of the column
 */

typedef std::tuple <int, std::string> ColumnKey;

/**
 * @type ColumnRowMap
 * @brief Maps a Column Key to column row structs
 */

typedef std::map < ColumnKey, ColumnRow *> ColumnRowMap;
        
class Catalog {

    public:
        Catalog(const std::string &basePath);
        ~Catalog();

        /**
         * @brief Writes in-memory representation of table table and column table to file 
         *
         * This method only writes to file if the catalog is "dirty", as specified
         * by the isDirty_ member variable.  It overwrites the existing catalog files.
         * Format is string (non-binary) with each field in the TableRow and ColumnRow
         * structs seperated by a tab, the rows themselves seperated by a newline
         */
        mapd_err_t writeCatalogToFile();

        /**
         * @brief Adds an empty table to the catalog 
         *
         * This method tries to add a new table to the Catalog (the table table),
         * returning an error if a table by the same name already exists.
         * It autoincrements the nextTableId_ counter so that the next table created
         * will have an id one higher.
         */
        mapd_err_t addTable(const std::string &tableName);

        /**
         * @brief Adds a table and a set of columns to the catalog 
         *
         * This method first determines whether the table and columns can be added
         * to the Catalog (i.e ensuring the table does not already exist and none 
         * of the column names are duplicates). Along with the table name it expects
         * a vector of ColumnRow structs, which it fills in the tableId and columnId
         * fields for.
         */

        mapd_err_t addTableWithColumns(const std::string &tableName, std::vector <ColumnRow *> & columns);


        /**
         * @brief Adds a column to the catalog for an already extant table 
         *
         * This method tries to add a new column to a table that is already in 
         * the catalog - equivalent to SQL's alter table add column command.
         * It returns an error if the table does not exist, or if the table does 
         * exist but a column with the same name as the column being inserted for
         * that table already exists
         */
        mapd_err_t addColumnToTable(const std::string &tableName, ColumnRow * columnRow);



        /**
         * @brief Removes a table and all of its associated columns from the catalog
         *
         * This method tries to remove the table specified by tableName from the 
         * Catalog.  It returns an error if no table by that name exists.  
         */
        mapd_err_t removeTable(const std::string &tableName);

        /**
         * @brief Removes a name-specified column from a given table from the catalog
         *
         * This method tries to remove the column specified by columnName from the
         * table specified by tableName, returning an error if no table by the given
         * table name exists for no column by the given column name exists for the
         * table specified. 
         */
        mapd_err_t removeColumnFromTable(const std::string &tableName, const std::string &columnName);

        /**
         * @brief Passes back via reference a ColumnRow struct for the column specified by table name and column name 
         *
         * This method first checks to see if the table and column specified by
         * the tableName and columnName parameters exist, returning an error if
         * they do not.  It then makes a copy of the ColumnRow struct representing
         * that column which returned via the columnRow parameter.  For now we
         * choose not to return the raw pointer as this could be invalidated by
         * the Catalog before the calling function can access it in a multithreaded
         * environment, although this might be a moot point if we never allow 
         * such query overlap in the first place
         */
        mapd_err_t getMetadataforColumn (const std::string &tableName, const std::string &columnName, ColumnRow &columnRow);


        /**
         * @brief Passes back via reference a vector of ColumnRow structs for the column specified by table name and column name 
         *
         * This method first checks to see if the table and columns specified by
         * the tableName and columnName parameters exist, returning an error if
         * they do not.  It then inserts into the vector of ColumnRow structs
         * passed as an argument to the function copies of all structs matching
         * the given columnName.
         */
        mapd_err_t getMetadataforColumns (const std::string &tableName, const std::vector<std::string> &columnNames,  std::vector <ColumnRow> &columnRows);


    private:

        /**
         * @brief Reads files representing state of table table and column table into in-memory representation
         *
         * This method first checks to see if the table file and column file exist,
         * only reading if they do.  It reads in data in the format specified by 
         * writeCatalogToFile().  It performs no error-checking as we assume the 
         * Catalog files are written by Catalog and represent the pre-valicated
         * in-memory state of the database metadata.
         */

        mapd_err_t readCatalogFromFile();

        std::string basePath_; /**< The OS file system path containing the catalog files. */
        TableRowMap tableRowMap_;
        ColumnRowMap columnRowMap_;
        int maxTableId_; /**< Serves as monotonically increasing counter to assign to each generated table. Increments on table creation but never decrements on deletion - so may have "gaps" */
        int maxColumnId_; /**< Serves as monotonically increasing counter to assign to each generated column. Increments on column creation but never decrements on deletion - so may have "gaps".  Right now we use a global columnId counter, making columnId a primary key for each column, but we may change this in the future so that each table has its own space for column keys. */
        bool isDirty_; /**< Specifies if the catalog has been modified in memory since the last flush to file - no need to rewrite file if this is false. */


};

#endif // CATALOG_H
