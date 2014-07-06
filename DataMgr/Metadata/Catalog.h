/**
 * @file    Catalog.h
 * @author  Todd Mostak <todd@map-d.com>
 * @brief   This file contains the class specification and related data structures for Catalog.
 *
 * This file contains the Catalog class specification. The Catalog class is responsible for storing metadata
 * about stored objects in the system (currently just relations).  At this point it does not take advantage of the 
 * database storage infrastructure, this likely will change in the future as the buildout continues. Although it persists
 * the metainfo on disk, at database startup it reads everything into in-memory dictionaries for fast access.
 *
 */

#ifndef CATALOG_H
#define CATALOG_H

#include <string>
#include <tuple>
#include <map>

#include "../../Shared/errors.h"

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
};

/**
 * @type TableRowMap
 * @brief Maps table names to pointers to table row structs 
 */

typedef std::map<string, TableRow *> TableRowMap;

/**
 * @type ColumnKey
 * @brief ColumnKey is composed of the integer tableId and the string name of the column
 */

typedef std::tuple<int, std::string> ColumnKey;

/**
 * @type ColumnRowMap
 * @brief Maps a Column Key to column row structs
 */

typedef std::map < ColumnKey, ColumnRow *> ColumnRowMap;
        
class Catalog {

    public:
        Catalog(const std::string &basePath);

        /**
         * @brief Writes in-memory representation of table table and column table to file 
         *
         * This method only writes to file if the catalog is "dirty", as specified
         * by the isDirty_ member variable.  It overwrites the existing catalog files.
         * Format is string (non-binary) with each field in the TableRow and ColumnRow
         * structs seperated by a tab, the rows themselves seperated by a newline
         */
        mapd_err_t writeCatalogToFile();



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
        bool isDirty_; /**< Specifies if the catalog has been modified in memory since the last flush to file - no need to rewrite file if this is false. */

};

#endif // CATALOG_H
