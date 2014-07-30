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
void test_writeReadFile(mapd_size_t blockSize, mapd_size_t nblocks);
void test_putGetBlock(mapd_size_t blockSizeArg, mapd_size_t nblocksArg);
void test_clearFreeBlock(mapd_size_t blockSizeArg, mapd_size_t nblocksArg);
void test_createChunk(mapd_size_t nblocks, mapd_size_t blockSizeArg);
void test_putGetChunk(mapd_size_t nblocks, mapd_size_t blockSizeArg);

int main(void) {
/*
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

    test_writeReadFile(4, 1);
    test_writeReadFile(3, 1);
    test_writeReadFile(2, 1);
    test_writeReadFile(1, 4);
    test_writeReadFile(2, 4);
    test_writeReadFile(1, 1);
    test_writeReadFile(1000, 1000);
    test_writeReadFile(10000, 10000);

    // don't uncomment this if you know what's good for you
    //test_writeReadFile(30000, 30000);

    test_putGetBlock(16, 16);
    test_putGetBlock(1, 1);
    test_putGetBlock(1000, 1000);
    test_putGetBlock(10000, 10000);

    // don't uncomment this if you know what's good for you
    // test_putGetBlock(30000, 30000);

    test_clearFreeBlock(8, 8);
    test_clearFreeBlock(128, 64);
    test_clearFreeBlock(1, 1);
    test_clearFreeBlock(1000, 1000);
    test_clearFreeBlock(10000, 10000);
    
    // don't uncomment this if you know what's good for you
    //test_clearFreeBlock(30000, 30000);
    //test_clearFreeBlock(100000, 100000);
*/
    test_createChunk(32, 8);
    test_putGetChunk(32, 8);
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

void test_writeReadFile(mapd_size_t blockSize, mapd_size_t nblocks) {
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
        PFAIL("test_writeReadFile() - error returned from writeFile writing uninitialized block");
    
    //initialize block1 with values
    for (int i = 0; i < blockSize; i++) {
        block1[i] = (mapd_byte_t) i;
    }

    err = fm.writeFile(*fInfo, 0, blockSize, (mapd_addr_t)block1);
    if (err != MAPD_SUCCESS)
        PFAIL("test_writeReadFile() - error returned from writeFile writing initialized block");
    
    // test read
    err = fm.readFile(*fInfo, 0, blockSize, (mapd_addr_t)buf1);
    if (err != MAPD_SUCCESS)
        PFAIL("test_writeReadFile() - error returned from readFile reading initialized block");
    
    loopError = false;
    for (int i = 0; i < blockSize; i++) {
        loopError = !(buf1[i] == block1[i]);
    }
    if (loopError) PFAIL("test_writeReadFile() - at least one of the read bytes is unequal to the written byte");
    else PPASS("test_writeReadFile() - all written bytes are read");

    //initialze block2 with values
    for (int i = 0; i < blockSize; i++) {
        block2[i] = rand() % 256;
    }

    //offset testing
    loopError = false;
    for (int i = 0; i < nblocks; i++) {

        //write blockSize bytes at position i*blockSIze
        err = fm.writeFile(*fInfo, i*blockSize, blockSize, (mapd_addr_t)block2);
        if (err != MAPD_SUCCESS)
            PFAIL("test_writeReadFile() - error returned from writeFile writing initialized block with offset");
        
        // read blockSize bytes at position i*blockSize
        err = fm.readFile(*fInfo, i*blockSize, blockSize, (mapd_addr_t)buf2);
        if (err != MAPD_SUCCESS)
            PFAIL("test_writeReadFile() - error returned from readFile reading initialized block");
        
        // make sure they all match!
        for (int i = 0; i < blockSize; i++) {
            loopError = !(buf1[i] == block1[i]);
        }

    } 
    if (loopError) PFAIL("test_writeReadFile() - offset testing: at least one of the read bytes is unequal to the written byte");
    else PPASS("test_writeReadFile() - offset testing: all written bytes are read");
}

void test_putGetBlock(mapd_size_t blockSizeArg, mapd_size_t nblocksArg) {
    bool loopError;
    mapd_err_t err;
    FileMgr fm(".");
    
    mapd_size_t blockSize = blockSizeArg;
    mapd_size_t nblocks = nblocksArg;
    
    FileInfo *fInfo = fm.createFile(blockSize, nblocks);

    mapd_byte_t block[blockSize];
    mapd_byte_t buf[blockSize];
    
    //initialze block with values
    for (int i = 0; i < blockSize; i++) {
         block[i] = (mapd_byte_t) rand() % 256;
    }

    int fileId = fInfo->fileId;

    // Insert blocks
    loopError = false;
    for (int i = 0; i < nblocks; ++i) {

        err = fm.putBlock(fileId, i, (mapd_addr_t) block);
        if (err != MAPD_SUCCESS) loopError = true;
    }
    if (loopError) 
        PFAIL("test_putGetBlock() - err returned by putBlock()");

    // Retrieve blocks
    loopError = false; 
    for (int i = 0; i < nblocks; ++i) {
        
        Block *b = fm.getBlock(fileId, i);
        if (!b) 
            PFAIL("test_putGetBlock() - getBlock returns a NULL block");

        // read the retrieved block
        size_t bytesRead = readBlock(fInfo->f, blockSize, i, (mapd_addr_t)buf, &err);

        // verify readBlock
        if (err != MAPD_SUCCESS || bytesRead != blockSize) {
            PFAIL("test_putGetBlock() - unsuccessful read");
        }

        // verify each byte in the read Block 
        for (int i = 0; i < blockSize; i++) {
            loopError = !(buf[i] == block[i]);
        }
        if (loopError) PFAIL("test_putGetBlock() - at least one retrieved byte does not match put byte");
        break;
    }
    if (!loopError) PPASS("test_putGetBlock() - all put bytes match retrieved bytes ");
}

// does the same as test_getPutBlock(), but checks that the block is clear upon retrieval
void test_clearFreeBlock(mapd_size_t blockSizeArg, mapd_size_t nblocksArg) {
    bool loopError;
    mapd_err_t err;
    FileMgr fm(".");
    
    mapd_size_t blockSize = blockSizeArg;
    mapd_size_t nblocks = nblocksArg;
    
    FileInfo *fInfo = fm.createFile(blockSize, nblocks);

    mapd_byte_t block[blockSize];
    mapd_byte_t buf[blockSize];
    
    //initialze block with values
    for (int i = 0; i < blockSize; i++) {
         block[i] = (mapd_byte_t) rand() % 256;
    }

    int fileId = fInfo->fileId;

    // Insert blocks
    for (int i = 0; i < nblocks; ++i) {
        err = fm.putBlock(fileId, i, (mapd_addr_t) block);
    }

    //clear blocks
    loopError = false;
    for (int i = 0; i < nblocks; ++i) {
        err = fm.clearBlock(fileId, i);
        if (err != MAPD_SUCCESS) 
            loopError = true;
    }
    if (loopError)
        PFAIL("test_clearFreeBlock() - error returned from clearBlock");

    // Retrieve blocks
    loopError = false; 
    for (int i = 0; i < nblocks; ++i) {
        
        Block *b = fm.getBlock(fileId, i);
        if (!b)

        loopError = (b->begin == b->end);
    }
    if (!loopError) PPASS("test_clearFreeBlock() - all blocks are clear");
    else PFAIL("test_clearFreeBlock() - some blocks are not clear");

    //time to free blocks

    //initialze block with values
    for (int i = 0; i < blockSize; i++) {
         block[i] = (mapd_byte_t) 1;
    }

    // Insert blocks
    for (int i = 0; i < nblocks; ++i) {
        err = fm.putBlock(fileId, i, (mapd_addr_t) block);
    }

    // free, free at last!
    loopError = false;
    for (int i = 0; i < nblocks; ++i) {
        err = fm.freeBlock(fileId, i);
        if (err != MAPD_SUCCESS) 
            loopError = true;
    }
    if (loopError)
        PFAIL("test_clearFreeBlock() - error returned from freeBlock");

    // are they really free- does the number of free blocks = original # of blocks?
    if (fInfo->available() == nblocks*blockSize) 
        PPASS("test_clearFreeBlock() - all blocks are freed");
    else {
        PFAIL("test_clearFreeBlock() - number of freed blocks does not equal number of blocks");
      //  printf("avail: %d, nblocks: %d\n", fInfo->available(), nblocks);
    }
}

void test_createChunk(mapd_size_t nblocks, mapd_size_t blockSizeArg) {
    mapd_err_t err;
    bool loopError = false;

    mapd_size_t blockSize1 = blockSizeArg;

    mapd_byte_t srcBuf[blockSize1];
    mapd_byte_t destBuf[blockSize1];
    FileMgr fm(".");

    int keyInt = 4;
    int epoch = 0;

    FileInfo *fileInfo1 = fm.createFile(blockSize1, nblocks);
    
    ChunkKey key;
    key.push_back(keyInt);
  //  printf("%d\n", key.back());

    int freeCount1 = fileInfo1->freeBlocks.size();\

    // creating a Chunk with the blockSize
    mapd_size_t sizeArg = blockSize1;
    Chunk* c = fm.createChunk(key, 0, blockSize1, NULL, 0);
 
    // verify that it actually returned
    if (c == NULL)
        PFAIL("createChunk returned a Null chunk");
    else PPASS("createChunk's returned Chunk exists");

    //@todo add support for creating chunk from source file.
}

void test_putGetChunk(mapd_size_t blockSizeArg, mapd_size_t nblocks) {
    mapd_err_t err;
    bool loopError = false;

    mapd_size_t blockSize1 = blockSizeArg;

    mapd_byte_t srcBufSmall[blockSize1];
    mapd_byte_t destBufSmall[blockSize1];
    FileMgr fm(".");

    int keyInt = 4;
    int epoch = 0;

    FileInfo *fileInfo1 = fm.createFile(blockSize1, nblocks);
    
    ChunkKey key;
    key.push_back(keyInt);
  //  printf("%d\n", key.back());

    int freeCount1 = fileInfo1->freeBlocks.size();\

    // creating a Chunk with the blockSize
    mapd_size_t sizeArg = blockSize1;
    Chunk* c = fm.createChunk(key, 0, blockSize1, NULL, 0);

    // initialize the buffer
    for (int i = 0; i < blockSize1; i++) {
        srcBufSmall[i] = rand() % 256;
    }

    Chunk* gottenRefChunk = fm.getChunkRef(key);
    if (gottenRefChunk == NULL)
        PFAIL("getChunkRef returned NULL");
    else PPASS("getChunkRef did not return NULL");

    err = fm.putChunk(key, blockSize1, srcBufSmall, epoch, blockSize1);
    if (err != MAPD_SUCCESS)
        PFAIL("putChunk returned error");
    else PPASS("putChunk did not return error");
  
    int freeCount2 = fileInfo1->freeBlocks.size();

    // verify that there is one less freeBlock
    if (freeCount1 != freeCount2 + 1)
        PFAIL("freeBlock size hasn't decreased by one");
    else PPASS("freeBlock size decreased by one");

    // put the chunk's content in the buffer
    Chunk* gottenChunk = fm.getChunk(key, destBufSmall);
    if (gottenChunk == NULL)
        PFAIL("getChunk returned NULL");
    else PPASS("getChunk did not return NULL");

    for (int i = 0; i < blockSize1; i++) {
        loopError = !(srcBufSmall[i] == destBufSmall[i]);
    }

    if (loopError)
        PFAIL("buffer filled during getChunk does not equal buffer inserted with putChunk");
    else PPASS("srcBuf and destBuf match");

    /* ----------------------------------------------------------------*/ 
    // serious testing: fill large buffers to capacity
    mapd_byte_t srcBuf[blockSize1*nblocks];
    mapd_byte_t destBuf[blockSize1*nblocks];

    // make another file just in case no space
    fm.createFile(blockSize1, nblocks);

    for (int i = 0; i < blockSize1*nblocks; i++) {
        srcBuf[i] = rand() % 256;
    }
 
    gottenRefChunk = fm.getChunkRef(key);
    if (gottenRefChunk == NULL)
        PFAIL("getChunkRef returned NULL");
    else PPASS("getChunkRef did not return NULL");

    err = fm.putChunk(key, blockSize1*nblocks, srcBuf, epoch);
    if (err != MAPD_SUCCESS)
        PFAIL("putChunk returned error");
    else PPASS("putChunk did not return error");

    // put the chunk's content in the buffer (no one puts ChunkKey in a corner)
    gottenChunk = fm.getChunk(key, destBuf);
    if (gottenChunk == NULL)
        PFAIL("getChunk returned NULL");
    else PPASS("getChunk did not return NULL");

    for (int i = 0; i < blockSize1*nblocks; i++) {
        loopError = !(srcBufSmall[i] == destBufSmall[i]);
    }

    if (loopError)
        PFAIL("buffer filled during getChunk does not equal buffer inserted with putChunk");
    else PPASS("srcBuf and destBuf match");


/*
    err = fm.deleteChunk(*c);
    if (err != MAPD_SUCCESS)
        PFAIL("deleteChunk returned error");
    else PPASS("deleteChunk did not return error");

    err = fm.deleteChunk(*gottenRefChunk);
    if (err != MAPD_SUCCESS)
        PFAIL("deleteChunk returned error");
    else PPASS("deleteChunk did not return error");

*/
    // we're going to create chunks of one multiblock 
    //Chunk* c = createChunk()
}

