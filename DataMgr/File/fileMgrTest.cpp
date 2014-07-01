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
#include "FileMgr.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

using namespace Testing;

// unit test function prototypes
bool test_createFile();
bool test_deleteFile();

int main(void) {

    test_createFile() ? 
        PPASS("createFile()") : PFAIL("createFile()"); 
    test_deleteFile() ? 
        PPASS("deleteFile()") : PFAIL("deleteFile()"); 

    printTestSummary();

	return EXIT_SUCCESS;
}

bool test_createFile() {
    mapd_err_t err;
    FileMgr fm(".");
    
    mapd_size_t blockSize = 32;
    mapd_size_t nblocks = 32;
    
    FileInfo *fInfo = fm.createFile(blockSize, nblocks, &err);
    
    if (!fInfo)
        return false;
    else {
        if (!fInfo->f
            || File::fileSize(fInfo->f) != blockSize * nblocks 
                || fInfo->blockSize != blockSize 
                    || fInfo->nblocks != nblocks)
            return false;
    }
    return true;
}

bool test_deleteFile() {
    mapd_err_t err;
    FileMgr fm(".");
    
    mapd_size_t blockSize = 32;
    mapd_size_t nblocks = 32;
    
    FileInfo *fInfo1;
    FileInfo *fInfo2;
    
    // create file
    fInfo1 = fm.createFile(blockSize, nblocks, &err);
    int fileId = fInfo1->fileId;
    
    // find the created file
    fInfo2 = fm.findFile(fileId, NULL);
    if (fInfo1 != fInfo2) // should point to the same object
        return false;
    
    // delete the created file
    err = fm.deleteFile(fileId);
    if (err != MAPD_SUCCESS)
        return false;

    return true;
}












