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
void test_FileInfo(const int nblocksArg);
//void test_FileInfo_size();

void test_FileMgr();
void test_createFile(mapd_size_t blockSizeArg, mapd_size_t nblocksArg);
void test_getFile(mapd_size_t blockSizeArg, mapd_size_t nblocksArg);
void test_deleteFile(mapd_size_t blockSizeArg, mapd_size_t nblocksArg);
void test_writeNreadFile(mapd_size_t blockSize, mapd_size_t nblocks);
void test_getBlock();
void test_freeBlock();

int main(void) {

	test_FileInfo(10);
    test_FileInfo(100);
    test_FileInfo(1000);
    test_FileInfo(10000);
	
    test_FileMgr();
    
    test_createFile(8, 8);
    test_createFile(128, 64);
    test_createFile(1, 1);
    test_createFile(1000, 1000);
    test_createFile(100000, 100000);

    test_getFile(8, 8);
    test_getFile(128, 64);
    test_getFile(1, 1);
    test_getFile(1000, 1000);
    test_getFile(100000, 100000);

    test_deleteFile(8, 8);
    test_deleteFile(1, 1);
    test_deleteFile(1000, 1000);
    test_deleteFile(100000, 100000);


    test_writeNreadFile(4, 1);
    test_writeNreadFile(3, 1);
    test_writeNreadFile(2, 1);
    test_writeNreadFile(1, 4);
    test_writeNreadFile(2, 4);
    test_writeNreadFile(1, 1);
    test_writeNreadFile(1000, 1000);
    test_writeNreadFile(100000, 100000);
    /*    PPASS("deleteFile()") : PFAIL("deleteFile()");*/
    /*test_getBlock() ?
        PPASS("getBlock()") : PFAIL("getBlock()"); 
    test_freeBlock() ? 
        PPASS("freeBlock()") : PFAIL("freeBlock()"); 
       */
    printTestSummary();

	return EXIT_SUCCESS;
}

void test_FileInfo(const int nblocksArg) {
	const int nblocks = nblocksArg;
	FileInfo fInfo(0, NULL, BLOCKSIZE, nblocks);
    bool loopError;

	// all blocks should have been created
	if (fInfo.blocks.size() != nblocks)
		PFAIL("test_FileInfo() size of blocks does not equal size parameter");
	Testing::pass++;

	// check size()
	if (fInfo.size() != nblocks * BLOCKSIZE)
		PFAIL("test_FileInfo() - size() incorrect");
	else Testing::pass++;

	// check available()
	if (fInfo.size() != fInfo.available())
		PFAIL("test_FileInfo() - available() incorrect");
	else Testing::pass++;

	// check used()
	if (fInfo.used() != 0)
        PFAIL("test_FileInfo() - used() incorrect");
	else Testing::pass++;

	// check the block size of the file
	if (fInfo.blockSize != BLOCKSIZE)
		PFAIL("test_FileInfo() - block size incorrect");
	else Testing::pass++;

    loopError = false;
	for (int i = 0; i < fInfo.blocks.size(); ++i) {
		// all blocks belong to file 0 and have an address of i*BLOCKSIZE bytes
		if (fInfo.blocks[i]->fileId != 0 || fInfo.blocks[i]->begin != i*BLOCKSIZE)
			loopError = true;
	}
    if (loopError) PFAIL("test_FileInfo() - at least one of fInfo's blocks belong to other file or have wrong size");
    else Testing::pass++;

    loopError = false;
	// all blocks are initially empty
	for (int i = 0; i < fInfo.blocks.size(); ++i) {
		if (fInfo.blocks[i]->end - fInfo.blocks[i]->begin != 0)
			loopError = true;
	}
    if (loopError) PFAIL("test_FileInfo() - at least one of fInfo's blocks are non empty");
	else Testing::pass++;

	// initially, all blocks are free
	if (fInfo.freeBlocks.size() != fInfo.blocks.size())
		PFAIL("test_FileInfo() - size of free blocks does not equal size of all blocks upon initialization");
	else Testing::pass++;

    PPASS("test_FileInfo() - all fields correct");

}

void test_FileMgr() {
	try {
		FileMgr *fMgr = new FileMgr(".");
		if (!fMgr)
			PFAIL("test_fileMgr() - fileManager object not created!");
		delete fMgr;
	} catch (const std::exception &e) {
		PFAIL("test_FileMgr() - fileManager exception thrown");
	}
    PPASS("test_FileMgr() - fileManager object created");
}

void test_createFile(mapd_size_t blockSizeArg, mapd_size_t nblocksArg) {
    FileMgr fm(".");
    
    mapd_size_t blockSize = blockSizeArg;
    mapd_size_t nblocks = nblocksArg;

    FileInfo *fInfo = fm.createFile(blockSize, nblocks);

    if (!fInfo)
        PFAIL("test_createFile() - fileInfo not generated from create File");

    else {
        if (!fInfo->f
            || fileSize(fInfo->f) != blockSize * nblocks 
                || fInfo->blockSize != blockSize 
                    || fInfo->nblocks != nblocks)
            PFAIL("test_createFile() - fileInfo generated, but erroneous");
        else PPASS("test_createFile() - fileInfo generated correctly");
    }
}

void test_getFile(mapd_size_t blockSizeArg, mapd_size_t nblocksArg) {
    FileMgr fm(".");

    mapd_size_t blockSize = blockSizeArg;
    mapd_size_t nblocks = nblocksArg;

    FileInfo *fInfo1 = fm.createFile(blockSize, nblocks);
    FileInfo *fInfo2 = fm.getFile(fInfo1->fileId);
    if (fInfo1 != fInfo2)
    	PFAIL("test_getFile() - fileInfo generated from createFile different from fileInfo from getFile");

    else PPASS("test_getFile() - fileInfo from getfile matches createFile");
}

void test_deleteFile(mapd_size_t blockSizeArg, mapd_size_t nblocksArg) {
    mapd_err_t err;
    FileMgr fm(".");
    
    mapd_size_t blockSize = blockSizeArg;
    mapd_size_t nblocks = nblocksArg;
    
    FileInfo *fInfo1;
    FileInfo *fInfo2;
    
    // create file
    fInfo1 = fm.createFile(blockSize, nblocks);
    int fileId = fInfo1->fileId;
    // fInfo1->print(true);
    
    // find the created file
    fInfo2 = fm.getFile(fileId);
    if (fInfo1 != fInfo2) // should point to the same object
        PFAIL("test_getFile() - fileInfo generated from createFile different from fileInfo from getFile");

    // delete the created file
    err = fm.deleteFile(fileId);
    if (err != MAPD_SUCCESS)
        PFAIL("test_deleteFile() - error returned from deleteFile");
	else PPASS("test_deleteFile() - no error returned from deleteFile");
	// @todo write this test
    //return true;
}

void test_writeNreadFile(mapd_size_t blockSize, mapd_size_t nblocks) {
    mapd_err_t err;
    FileMgr fm(".");
    bool loopError;

    mapd_byte_t block0[blockSize];
    mapd_byte_t block1[blockSize];
    mapd_byte_t block2[blockSize];
    mapd_byte_t buf1[blockSize];
    mapd_byte_t buf2[blockSize];

    FileInfo *fInfo = fm.createFile(blockSize, nblocks);

    //write an empty block into the file
    err = fm.writeFile(*fInfo, 0, blockSize, (mapd_addr_t)block0);
    if (err != MAPD_SUCCESS)
        PFAIL("test_writeNreadFile() - error returned from writeFile writing uninitialized block");
    
    //initialze block1 with values
    for (int i = 0; i < blockSize; i++) {
        block1[i] = (mapd_byte_t) i;
    }

    err = fm.writeFile(*fInfo, 0, blockSize, (mapd_addr_t)block1);
    if (err != MAPD_SUCCESS)
        PFAIL("test_writeNreadFile() - error returned from writeFile writing initialized block");
    
    // test read
    err = fm.readFile(*fInfo, 0, blockSize, (mapd_addr_t)buf1);
    if (err != MAPD_SUCCESS)
        PFAIL("test_writeNreadFile() - error returned from readFile reading initialized block");
    
    loopError = false;
    for (int i = 0; i < blockSize; i++) {
        loopError = !(buf1[i] == block1[i]);
    }
    if (loopError) PFAIL("test_writeNreadFile() - at least one of the read bytes is unequal to the written byte");
    else PPASS("test_writeNreadFile() - all written bytes are read");

/* @todo: 
    //offset testing
    for (int i = 0; i < nblocks; i++) {
        err = fm.writeFile(*fInfo, i*blockSize, blockSize, (mapd_addr_t)block1);
        if (err != MAPD_SUCCESS)
            PFAIL("test_writeNreadFile() - error returned from writeFile writing initialized block with offset");
        
        // test read
        err = fm.readFile(*fInfo, 1, blockSize, (mapd_addr_t)buf1);
        if (err != MAPD_SUCCESS)
            PFAIL("test_writeNreadFile() - error returned from readFile reading initialized block");
        
        loopError = false;
        for (int i = 0; i < blockSize; i++) {
            loopError = !(buf1[i] == block1[i]);
        }
        if (loopError) PFAIL("test_writeNreadFile() - at least one of the read bytes is unequal to the written byte");
        else PPASS("test_writeNreadFile() - all written bytes are read");
   } */
}
/*
void test_getBlock(mapd_size_t blockSizeArg, mapd_size_t nblocksArg) {
    mapd_err_t err;
    FileMgr fm(".");
    
    mapd_size_t blockSize = blockSizeArg;
    mapd_size_t nblocks = nblocksArg;
    
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
/*
bool test_freeBlock() {
    //@todo write this test
    return false;
}
*/
