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

/**
 * @type TableRow
 * @brief specifies the content in-memory of a row in the table metadata table
 * 
 * A TableRow type currently includes only the table name and the tableId (zero-based) that it maps to. Other metadata could be added in the future
 */

struct TableRow {
    std::string tableName; /**< tableId is the primary key to access rows in the table table -must bew unique */
    int tableId; /**< tableId starts at 0 for valid tables. */
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





    private:
        std::string basePath_;
        TableRowMap tableRowMap_;
        ColumnRowMap columnRowMap_;

        //std::map <std::string, TableRow *> tableRowMap_;
        //std::map < std::tuple<int, std::string>, ColumnRow *> columnRowMap_;


        





};

#endif // CATALOG_H
