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
void test_FileInfo_Constructor();
void test_FileInfo_Destructor();
void test_FileMgr_Constructor();
void test_FileMgr_createFile();
void testFileMgr_readAndWriteState();

int main(void) {
    
    test_FileInfo_Constructor();
    test_FileInfo_Destructor();
    
    test_FileMgr_Constructor();
    test_FileMgr_createFile();
    testFileMgr_readAndWriteState();
    
    printTestSummary();
    
    return EXIT_SUCCESS;
}

void test_FileInfo_Constructor() {
    int fileId = 0;
    mapd_size_t blockSize = 32;
    mapd_size_t nblocks = 32;
    mapd_err_t err = MAPD_SUCCESS;
    
    // File creation
    FILE *f = create(fileId, blockSize, nblocks, &err);
    if (f == NULL || err != MAPD_SUCCESS) {
        PFAIL("File creation");
        return;
    }
    
    // Constructor
    FileInfo fInfo(fileId, f, blockSize, nblocks);
    if (fInfo.fileId != fileId || fInfo.blockSize != blockSize || fInfo.nblocks != nblocks) {
        PFAIL("Member variable initialization");
        return;
    }
    
    // available()
    if (fInfo.available() != blockSize * nblocks) {
        PFAIL("Free blocks initializtion");
        return;
    }
    
    // used()
    if (fInfo.used() != 0) {
        PFAIL("Used bytes not initialized to 0");
        return;
    }
    
    // size
    if (blockSize * nblocks != fInfo.size()) {
        PFAIL("incorrect size");
        return;
    }
    
    // remove the file
    /*if (removeFile(".", fileId + ".mapd") != MAPD_SUCCESS) {
     PFAIL("removeFile");
     return;
     }*/
    
    // Conclusion
    PPASS("Constructor");
}

void test_FileInfo_Destructor() {
    int fileId = 0;
    mapd_size_t blockSize = 32;
    mapd_size_t nblocks = 32;
    mapd_err_t err = MAPD_SUCCESS;
    
    // File creation
    FILE *f = create(fileId, blockSize, nblocks, &err);
    if (f == NULL || err != MAPD_SUCCESS) {
        PFAIL("File creation");
        return;
    }
    
    // Constructor
    FileInfo *fInfo = new FileInfo(fileId, f, blockSize, nblocks);
    if (fInfo->fileId != fileId || fInfo->blockSize != blockSize || fInfo->nblocks != nblocks) {
        PFAIL("Member variable initialization");
        return;
    }
    
    // Destructor
    delete fInfo;
    // @todo this unit test needs more work
    
    // Conclusion
    PPASS("Destructor");
}

void test_FileMgr_Constructor() {
    std::string basePath = ".";
    FileMgr fm(basePath);
    
    // basePath
    if (fm.basePath() != basePath) {
        PFAIL("Incorrect base path");
        return;
    }
    
    // Conclusion
    PPASS("Constructor");
}

void test_FileMgr_createFile() {
    std::string basePath = ".";
    FileMgr fm(basePath);
    
    // create the file
    int blockSize = 16;
    int nblocks = 16;
    FileInfo *fInfo;
    
    if ((fInfo = fm.createFile(blockSize, nblocks)) == NULL) {
        PFAIL("Failure creating file");
    }
    
    
    // Conclusion 
    PPASS("createFile");   
}

void testFileMgr_readAndWriteState() {
    std::string basePath = ".";
    FileMgr fm(basePath);
    
    // clear metadata stored in Postgres database
    fm.clearState();
    
    // create the file
    int blockSize = 16;
    int nblocks = 16;
    FileInfo *fInfo;
    
    if ((fInfo = fm.createFile(blockSize, nblocks)) == NULL) {
        PFAIL("Failure creating file");
    }

    // Conclusion
    PPASS("readAndWriteState");
}








