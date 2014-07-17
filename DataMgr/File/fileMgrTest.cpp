/**
 * @file fileMgrTest.cpp
 * @author Steven Stewart <steve@map-d.com>
 *
 * This program is used to test the FileMgr class.
 *
 * @see FileMgr
 */
#include <cstdio>
#include <cstdlib>
#include <exception>
#include "FileMgr.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

using namespace Testing;
using namespace File_Namespace;

#define BLOCKSIZE 32

// unit test function prototypes
bool test_FileInfo();
bool test_FileInfo_size();

bool test_FileMgr();
bool test_createFile();
bool test_deleteFile();
bool test_getFile();
bool test_getBlock();
bool test_freeBlock();

int main(void) {

	test_FileInfo() ?
	    PPASS("FileInfo()") : PFAIL("FileInfo()");
	test_FileMgr() ?
		PPASS("FileMgr()") : PFAIL("FileMgr()");
    test_createFile() ?
        PPASS("FileMgr::createFile()") : PFAIL("FileMgr::createFile()");
    test_getFile() ?
    	PPASS("FileMgr::getFile()") : PFAIL("FileMgr::getFile()");
    /*test_deleteFile() ?
        PPASS("deleteFile()") : PFAIL("deleteFile()");*/
    /*test_getBlock() ?
        PPASS("getBlock()") : PFAIL("getBlock()"); 
    test_freeBlock() ? 
        PPASS("freeBlock()") : PFAIL("freeBlock()"); 
       */
    printTestSummary();

	return EXIT_SUCCESS;
}

bool test_FileInfo() {
	const int nblocks = 10;
	FileInfo fInfo(0, NULL, BLOCKSIZE, nblocks);

	// all blocks should have been created
	if (fInfo.blocks.size() != nblocks)
		return false;
	Testing::pass++;

	// check size()
	if (fInfo.size() != nblocks * BLOCKSIZE)
		return false;
	Testing::pass++;

	// check available()
	if (fInfo.size() != fInfo.available())
		return false;
	Testing::pass++;

	// check used()
	if (fInfo.used() != 0)
		return false;
	Testing::pass++;

	// check the block size of the file
	if (fInfo.blockSize != BLOCKSIZE)
		return false;
	Testing::pass++;

	for (int i = 0; i < fInfo.blocks.size(); ++i) {
		// all blocks belong to file 0 and have an address of i*BLOCKSIZE bytes
		if (fInfo.blocks[i]->fileId != 0 || fInfo.blocks[i]->begin != i*BLOCKSIZE)
			return false;
	}
	Testing::pass++;

	// all blocks are initially empty
	for (int i = 0; i < fInfo.blocks.size(); ++i) {
		if (fInfo.blocks[i]->end - fInfo.blocks[i]->begin != 0)
			return false;
	}
	Testing::pass++;

	// initially, all blocks are free
	if (fInfo.freeBlocks.size() != fInfo.blocks.size())
		return false;
	Testing::pass++;

	return true;
}

bool test_FileMgr() {
	try {
		FileMgr *fMgr = new FileMgr(".");
		if (!fMgr)
			return false;
		delete fMgr;
	} catch (const std::exception &e) {
		return false;
	}
	return true;
}

bool test_createFile() {
    FileMgr fm(".");
    
    mapd_size_t blockSize = 8;
    mapd_size_t nblocks = 8;

    FileInfo *fInfo = fm.createFile(blockSize, nblocks);

    if (!fInfo)
        return false;
    else {
        if (!fInfo->f
            || fileSize(fInfo->f) != blockSize * nblocks 
                || fInfo->blockSize != blockSize 
                    || fInfo->nblocks != nblocks)
            return false;
    }
    return true;
}

bool test_getFile() {
    FileMgr fm(".");

    mapd_size_t blockSize = 8;
    mapd_size_t nblocks = 8;

    FileInfo *fInfo1 = fm.createFile(blockSize, nblocks);
    FileInfo *fInfo2 = fm.getFile(fInfo1->fileId);
    if (fInfo1 != fInfo2)
    	return false;

	return true;
}

bool test_deleteFile() {
    /*mapd_err_t err;
    FileMgr fm(".");
    
    mapd_size_t blockSize = 8;
    mapd_size_t nblocks = 8;
    
    FileInfo *fInfo1;
    FileInfo *fInfo2;
    
    // create file
    fInfo1 = fm.createFile(blockSize, nblocks, &err);
    int fileId = fInfo1->fileId;
    // fInfo1->print(true);
    
    // find the created file
    fInfo2 = fm.getFile(fileId, NULL);
    if (fInfo1 != fInfo2) // should point to the same object
        return false;
    
    // delete the created file
    err = fm.deleteFile(fileId);
    if (err != MAPD_SUCCESS)
        return false;
	*/
	// @todo write this test
    return true;
}
/*
bool test_getBlock() {
    mapd_err_t err;
    FileMgr fm(".");
    
    mapd_size_t blockSize = 8;
    mapd_size_t nblocks = 8;
    
    FileInfo *fInfo1;
    FileInfo *fInfo2;
    
    // create file
    fInfo1 = fm.createFile(blockSize, nblocks, &err);
    int fileId = fInfo1->fileId;
    
    // Retrieve blocks
    for (int i = 0; i < nblocks; ++i) {
        BlockAddr *bAddr = fm.getBlock(fileId, i, &err);
        // printf("%u\n", bInfo->blk.blockAddr);
        if (err != MAPD_SUCCESS)
            return false;
        else if (bInfo->blk.blockAddr != i*blockSize)
            return false;
    }
    
    return true;
}

bool test_freeBlock() {
    //@todo write this test
    return false;
}
*/
