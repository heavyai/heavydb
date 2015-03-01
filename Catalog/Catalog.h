/**
 * @file    Catalog.h
 * @author  Todd Mostak <todd@map-d.com>, Wei Hong <wei@map-d.com>
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
#include <map>
#include <list>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <cstdint>

#include "../SqliteConnector/SqliteConnector.h"
#include "TableDescriptor.h"
#include "ColumnDescriptor.h"
#include "DictDescriptor.h"
#include "../DataMgr/DataMgr.h"

namespace Catalog_Namespace {


/**
 * @type TableDescriptorMap
 * @brief Maps table names to pointers to table descriptors allocated on the
 * heap 
 */

typedef std::map<std::string, TableDescriptor *> TableDescriptorMap;

typedef std::map<int, TableDescriptor *> TableDescriptorMapById;

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

typedef std::tuple <int, int> ColumnIdKey;
typedef std::map < ColumnIdKey, ColumnDescriptor *> ColumnDescriptorMapById;

typedef std::map < int, DictDescriptor *> DictDescriptorMapById;
        
/*
 * @type UserMetadata
 * @brief metadata for a mapd user
 */
struct UserMetadata {
	UserMetadata(int32_t u, const std::string &n, const std::string &p, bool s) : userId(u), userName(n), passwd(p), isSuper(s) {}
	UserMetadata() {}
	int32_t userId;
	std::string userName;
	std::string passwd;
	bool isSuper;
};

/*
 * @type DBMetadata
 * @brief metadata for a mapd database
 */
struct DBMetadata {
	int32_t dbId;
	std::string dbName;
	int32_t dbOwner;
};

/* database name for the system database */
#define MAPD_SYSTEM_DB "mapd"
/* the mapd root user */
#define MAPD_ROOT_USER "mapd"
#define MAPD_ROOT_USER_ID 0
#define MAPD_ROOT_USER_ID_STR "0"
#define MAPD_ROOT_PASSWD_DEFAULT "HyperInteractive"

/**
 * @type Catalog
 * @brief class for a per-database catalog.  also includes metadata for the
 * current database and the current user.
 */

class Catalog {

    public:
				Catalog(const std::string &basePath, const std::string &dbname, Data_Namespace::DataMgr &dataMgr, bool is_initdb);

        /**
         * @brief Constructor - takes basePath to already extant
         * data directory for writing
         * @param basePath directory path for writing catalog 
				 * @param dbName name of the database
				 * @param fragmenter Fragmenter object
         * metadata - expects for this directory to already exist
         */

        Catalog(const std::string &basePath, const UserMetadata &curUser, const DBMetadata &curDB, Data_Namespace::DataMgr &dataMgr);

        /**
         * @brief Destructor - deletes all
         * ColumnDescriptor and TableDescriptor structures 
         * which were allocated on the heap and writes
         * Catalog to Sqlite
         */
        virtual ~Catalog();

				void createTable(TableDescriptor &td, const std::list<ColumnDescriptor> &columns);
				void dropTable(const TableDescriptor *td);

        /**
         * @brief Returns a pointer to a const TableDescriptor struct matching
         * the provided tableName
         * @param tableName table specified column belongs to
         * @return pointer to const TableDescriptor object queried for or nullptr if it does not exist. 
         */

        const TableDescriptor * getMetadataForTable (const std::string &tableName) const;
        const TableDescriptor * getMetadataForTable (int tableId) const;

        const ColumnDescriptor * getMetadataForColumn(int tableId, const std::string &colName) const;
        const ColumnDescriptor * getMetadataForColumn(int tableId, int columnId) const;

        /**
         * @brief Returns a list of pointers to constant ColumnDescriptor structs for all the columns from a particular table 
         * specified by table id
         * @param tableId table id we want the column metadata for
         * @return list of pointers to const ColumnDescriptor structs - one
         * for each and every column in the table
         *
         */

         std::list <const ColumnDescriptor *> getAllColumnMetadataForTable(const int tableId) const;

				 std::list<const TableDescriptor *> getAllTableMetadata() const;
         const UserMetadata &get_currentUser() const { return currentUser_; }
         void set_currentUser(const UserMetadata &user) { currentUser_ = user; }
         const DBMetadata &get_currentDB() const { return currentDB_; }
         void set_currentDB(const DBMetadata &db) { currentDB_ = db; }
				 Data_Namespace::DataMgr &get_dataMgr() const { return dataMgr_; }

         const DictDescriptor *getMetadataForDict(int dictId) const;

    protected:
        void buildMaps();
        void addTableToMap(TableDescriptor &td, const std::list<ColumnDescriptor> &columns, const std::list<DictDescriptor> &dicts);
        void removeTableFromMap(const std::string &tableName, int tableId);
				void instantiateFragmenter(TableDescriptor *td) const;
				void getAllColumnMetadataForTable(const TableDescriptor *td, std::list<const ColumnDescriptor *> &colDescs) const;

        std::string basePath_; /**< The OS file system path containing the catalog files. */
        TableDescriptorMap tableDescriptorMap_;
        TableDescriptorMapById tableDescriptorMapById_;
        ColumnDescriptorMap columnDescriptorMap_;
        ColumnDescriptorMapById columnDescriptorMapById_;
        DictDescriptorMapById dictDescriptorMapById_;
        SqliteConnector sqliteConnector_;
        UserMetadata currentUser_;
        DBMetadata currentDB_;
				Data_Namespace::DataMgr &dataMgr_;
};

/*
 * @type SysCatalog
 * @brief class for the system-wide catalog, currently containing user and database metadata
 */
class SysCatalog : public Catalog {
	public:
		SysCatalog(const std::string &basePath, Data_Namespace::DataMgr &dataMgr, bool is_initdb = false) : Catalog(basePath, MAPD_SYSTEM_DB, dataMgr, is_initdb) {}
		virtual ~SysCatalog() {};
		void initDB();
		void createUser(const std::string &name, const std::string &passwd, bool issuper);
		void dropUser(const std::string &name);
		void alterUser(const std::string &name, const std::string *passwd, bool *is_superp);
		void createDatabase(const std::string &dbname, int owner);
		void dropDatabase(const std::string &dbname);
		bool getMetadataForUser(const std::string &name, UserMetadata &user);
		bool getMetadataForDB(const std::string &name, DBMetadata &db);
		std::list<DBMetadata> getAllDBMetadata();
		std::list<UserMetadata> getAllUserMetadata();
};


} // Catalog_Namespace

#endif // CATALOG_H
