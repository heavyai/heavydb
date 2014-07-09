/**
 * @file Database.h
 * @author Steven Stewart
 *
 */
#ifndef _DATABASE_H
#define _DATABASE_H

// definitions
#define CONFIG_HOST_MEM (1024*1048576)

// forward declarations
class FileMgr;
class BufferMgr;

namespace Database_Namespace {
/**
 * Database class
 * @todo description of this class
 */
class Database {

public:
	Database();
	virtual ~Database();

private:
	FileMgr *fm_;
	BufferMgr *bm_;

	Database(const Database& copy);
	Database& operator =(const Database&);
};

}

#endif // _DATABASE_H

