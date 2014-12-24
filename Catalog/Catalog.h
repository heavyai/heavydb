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

//@todo Be able to look up tables by schema and possibily database name (i.e.
//db_name.schema_name.table_name
//
#ifndef CATALOG_H
#define CATALOG_H



#include <string>
#include <tuple>
#include <map>
#include <vector>
#include <utility>
#include <boost/lexical_cast.hpp>

#include "../../Shared/errors.h"
#include "../../Shared/types.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "TableDescriptor.h"
#include "ColumnDescriptor.h"

namespace Catalog_Namespace {


/**
 * @type TableDescriptorMap
 * @brief Maps table names to pointers to table descriptors allocated on the
 * heap 
 */

typedef std::map<std::string, TableDescriptor *> TableDescriptorMap;

/**
 * @type ColumnKey
 * @brief ColumnKey is composed of the integer tableId and the string name of the column
 */

typedef std::tuple <int, std::string> ColumnKey;

/**
 * @type ColumnDescriptorMap
 * @brief Maps a Column Key to column descriptors allocated on the heap
 */

typedef std::map < ColumnKey, ColumnDescriptor *> ColumnDescriptorMap;
        
/**
 * @type Catalog
 * @brief Serves as the system catalog.  Currently can use
 * either ASCII files to store the Table and Column Tables
 * or Sqlite (default)
 */

class Catalog {

    public:
        /**
         * @brief Constructor - takes basePath to already extant
         * data directory for writing
         * @param basePath directory path for writing catalog 
         * metadata - expects for this directory to already exist
         */

        Catalog(const std::string &basePath);

        /**
         * @brief Destructor - deletes all
         * ColumnDescriptor and TableDescriptor structures 
         * which were allocated on the heap and writes
         * Catalog to Sqlite
         */
        ~Catalog();

        /**
         * @brief Writes in-memory representation of table table and column table to file 
         *
         * This method only writes to file if the catalog is "dirty", as specified
         * by the isDirty_ member variable.  It overwrites the existing catalog files.
         * Format is string (non-binary) with each field in the TableDescriptor and ColumnDescriptor
         * structs seperated by a tab, the rows themselves seperated by a newline
         */
        void writeCatalogToFile();

        /**
         * @brief Adds an empty table to the catalog 
         * @param tableName name of table to be added
         * @return integer tableId of created table
         *
         * This method tries to add a new table to the Catalog (the table table),
         * - throwing a runtime_error if a table by the same name already exists.
         * It autoincrements the nextTableId_ counter so that the next table created
         * will have an id one higher.
         */
        int addTable(const std::string &tableName);

        /**
         * @brief Adds a table and a set of columns to the catalog 
         * @param tableName name of table to be added
         * @param columns vector of ColumnDescriptors
         * that should have been
         * instanciated with second (partial)
         * constrructor. tableId and columnId will
         * be populated by Catalog for each.
         *
         * @return integer tableId of created table 
         *
         * This method first determines whether the table and columns can be added
         * to the Catalog (i.e ensuring the table does not already exist and none 
         * of the column names are duplicates). Along with the table name it expects
         * a vector of ColumnDescriptor structs, which it fills in the tableId and columnId
         * fields for.
         *
         * Expects that ColumnDescriptor structs are initialized with second
         * constructor (without tableId and columnId) 
         *
         * Called by SQL DDL CREATE TABLE
         *
         * @see ColumnDescriptor
         */

        int addTableWithColumns(const std::string &tableName, const std::vector <ColumnDescriptor> &columns);


        /**
         * @brief Adds a column to the catalog for an already extant table 
         *
         * @param ColumnDescriptor structure instanciated with second
         * (partial) constructor.  tableID and columnId will be populated
         * by Catalog.  
         *
         * This method tries to add a new column to a table that is already in 
         * the catalog - equivalent to SQL's alter table add column command.
         * It throws a runtime_error if the table does not exist, or if the table does 
         * exist but a column with the same name as the column being inserted for
         * that table already exists
         *
         * Called by SQL DDL ALTER TABLE ADD COLUMN
         *
         * @see ColumnDescriptor
         */
        void addColumnToTable(const std::string &tableName, const ColumnDescriptor &columnDescriptor);

        /**
         * @brief Removes a table and all of its associated columns from the catalog
         * @param tableName name of table to be removed from catalog
         *
         * This method tries to remove the table specified by tableName from the 
         * Catalog.  It thrwos a runtime_error if no table by that name exists.  
         *
         * Called by SQL DDL DROP TABLE
         */

        void removeTable(const std::string &tableName);

        /**
         * @brief Removes a name-specified column from a given table from the catalog
         * @param tableName table specified column belongs to
         * @param columnName name of column to be removed
         *
         * This method tries to remove the column specified by columnName from the
         * table specified by tableName, throwing a runtime error if no table by the given
         * table name exists for no column by the given column name exists for the
         * table specified. 
         *
         * Called by SQL DDL ALTER TABLE DROP COLUMN
         */

        void removeColumnFromTable(const std::string &tableName, const std::string &columnName);

        /**
         * @brief Returns a pointer to a const TableDescriptor struct matching
         * the provided tableName
         * @param tableName table specified column belongs to
         * @return pointer to const TableDescriptor object queried for
         * 
         * Throws a runtime error if the specified table does not exist
         */

        const TableDescriptor * getMetadataForTable (const std::string &tableName) const;

        /**
         * @brief Returns a pointer to a const ColumnDescriptor struct
         * for the column specified by table name and column name 
         * @param tableName table specified column belongs to
         * @param columnName name of column we want metadata for
         * @return pointer to const ColumnDescriptor for desired column 
         *
         * This method first checks to see if the table and column specified by
         * the tableName and columnName parameters exist, throwing an error if
         * they do not.  It then returns a pointer to the const desired ColumnDescriptor
         * struct. It is const because this points to the actual struct stored
         * on the heap by the Catalog.
         */

        const ColumnDescriptor * getMetadataForColumn (const std::string &tableName, const std::string &columnName) const;

        /**
         * @brief Returns a vector of pointers to constant ColumnDescriptor structs for 
         * the specified columns from a single table
         * @param tableName table specified columns belong to
         * @param columnNames vector of names of columns we want
         * @return vector of pointers to const ColumnDescriptor structs  that
         * match query
         *
         * This method first checks to see if the table and columns specified by
         * the tableName and columnName parameters exist, throwing a runtime error if 
         * they do not.  It then returns a vector of pointers to const
         * ColumnDescriptor structs matching the provided columnNames
         */

        std::vector <const ColumnDescriptor *> getMetadataForColumns (const std::string &tableName, const std::vector<std::string> &columnNames) const;

        /**
         * @brief Returns a vector of pointers to constant ColumnDescriptor structs for the specified columns from multiple tables
         * @param tableNames vector of table names columns can belong to
         * @param columnNames vector of pairs of names of columns we want,
         * signifying tableName.columnName.  tableName in the pair can be left
         * blank ("") if it is not known - the method will attempt to
         * disambiguate
         * @return vector of pointers to const ColumnDescriptor structs  that
         * match query
         *
         * This method expects a vector of table names and a vector of column
         * name "pairs", specifying tableName.columnName.  If tableName is not
         * given (i.e. left as ""), it will try to disambiguate, throwing an
         * error if it cannot. 
         */

         std::vector <const ColumnDescriptor *> getMetadataForColumns(const std::vector <std::string>  &tableNames, const std::vector <std::pair <std::string, std::string> > &columnNames) const;

        /**
         * @brief Returns a vector of pointers to constant ColumnDescriptor structs for all the columns from a particular table 
         * @param tableName table name we want the column metadata for
         * @return vector of pointers to const ColumnDescriptor structs - one
         * for each and every column in the table
         *
         * Used for select * queries
         */

         std::vector <const ColumnDescriptor *> getAllColumnMetadataForTable(const std::string &tableName) const;

        /**
         * @brief Returns a vector of pointers to constant ColumnDescriptor structs for all the columns from a particular table 
         * specified by table id
         * @param tableId table id we want the column metadata for
         * @return vector of pointers to const ColumnDescriptor structs - one
         * for each and every column in the table
         *
         * Called internally by getAllColumnMetadataForTable(const string tableName)
         */

         std::vector <const ColumnDescriptor *> getAllColumnMetadataForTable(const int tableId) const;

    private:

        
        inline std::string getTypeName(mapd_data_t type) {
            switch (type) {
                case INT_TYPE:
                    return "int";
                    break;
                case FLOAT_TYPE:
                    return "float";
                    break;
                case BOOLEAN_TYPE:
                    return "bool";
                    break;
            }
        }

        inline mapd_data_t getTypeFromString(const std::string &typeName) {
            if (typeName == "int") {
                return INT_TYPE;
            }
            else if (typeName == "float") {
                return FLOAT_TYPE;
            }
            else if (typeName == "bool") {
                return BOOLEAN_TYPE;
            }
            return INT_TYPE;
        }
        void createStateTableIfDne();

        /**
         * @brief Reads files representing state of table table and column table into in-memory representation
         *
         * This method first checks to see if the table file and column file exist,
         * only reading if they do.  It reads in data in the format specified by 
         * writeCatalogToFile().  It performs no error-checking as we assume the 
         * Catalog files are written by Catalog and represent the pre-valicated
         * in-memory state of the database metadata.
         */

        void readCatalogFromFile();

        void readState();
        void writeState();

        std::string basePath_; /**< The OS file system path containing the catalog files. */
        TableDescriptorMap tableDescriptorMap_;
        ColumnDescriptorMap columnDescriptorMap_;
        SqliteConnector sqliteConnector_;
        int maxTableId_; /**< Serves as monotonically increasing counter to assign to each generated table. Increments on table creation but never decrements on deletion - so may have "gaps" */
        int maxColumnId_; /**< Serves as monotonically increasing counter to assign to each generated column. Increments on column creation but never decrements on deletion - so may have "gaps".  Right now we use a global columnId counter, making columnId a primary key for each column, but we may change this in the future so that each table has its own space for column keys. */
        bool isDirty_; /**< Specifies if the catalog has been modified in memory since the last flush to file - no need to rewrite file if this is false. */


};

} // Catalog_Namespace

#endif // CATALOG_H
