/**
 *  File:       fileTest.cpp
 *  Author(s):  steve@map-d.com
 * 
 */
#include <cstdio>
#include <cstdlib>
#include "File.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

// function prototypes
bool test_open();
bool test_close();
bool test_write();
bool test_read();
bool test_append();
bool test_writeBlock();
bool test_readBlock();
bool test_appendBlock();

#define BLOCKSIZE 10
File *f = new File(BLOCKSIZE);

char block0[BLOCKSIZE];
char block1[BLOCKSIZE];

int main() {

    // populate block arrays used for testing
    for (int i = 0; i < BLOCKSIZE; i++) {
        block0[i] = i;
        block1[i] = BLOCKSIZE - i;
    }

    // call unit tests
    test_open() ? 
        PPASS("open()") : PFAIL("open()"); 
    test_close() ? 
        PPASS("close()") : PFAIL("close()"); 
    test_read() ? 
        PPASS("read()") : PFAIL("read()"); 
    test_readBlock() ? 
        PPASS("readBlock()") : PFAIL("readBlock()"); 
    test_write() ? 
        PPASS("write()") : PFAIL("write()"); 
    test_writeBlock() ? 
        PPASS("writeBlock()") : PFAIL("writeBlock()"); 
    test_append() ?
        PPASS("append()") : PFAIL("append()"); 
    test_appendBlock() ? 
        PPASS("appendBlock()") : PFAIL("appendBlock()"); 
    
    delete f;   
    
    printTestSummary();
    
    return 0;
}

bool test_open() {
    if (f->open("test.db", true) != MAPD_SUCCESS)
        return false;
    return (f->close() == MAPD_SUCCESS);
}

bool test_close() {
    f->open("test.db", true);
    return (f->close() == MAPD_SUCCESS);
}

bool test_read() {
    int i, j;
    char buf[BLOCKSIZE];

    // create a file for testing
    if (f->open("test.db", true) != MAPD_SUCCESS)
        return false;

    // write block0 to the file
    if (f->write(0, BLOCKSIZE, &block0) != MAPD_SUCCESS) {
        f->close();
        return false;
    }
    
    // write block1 to the file
    if (f->append(BLOCKSIZE, &block1) != MAPD_SUCCESS) {
        f->close();
        return false;
    }   

    // read block0 from the file and verify its contents
    if (f->read(0, BLOCKSIZE, &buf) != MAPD_SUCCESS) {
        f->close();
        return false;
    }
    for (i = 0; i < BLOCKSIZE; ++i) {
        if (block0[i] != buf[i]) {
            f->close();
            return false;
        }
    }

    // read block1 from the file and verify its contents
    if (f->read(BLOCKSIZE, BLOCKSIZE, &buf) != MAPD_SUCCESS) {
        f->close();
        return false;
    }
    for (i = 0, j = BLOCKSIZE; i < BLOCKSIZE; ++i, ++j) {
        if (block1[i] != buf[i]) {
            f->close();
            return false;
        }
    }

    return (f->close() == MAPD_SUCCESS);
}

bool test_readBlock() {
    return false;
}

bool test_write() {
     // create a file for testing
    if (f->open("test.db", true) != MAPD_SUCCESS)
        return false;
    
    if (f->fileSize() != 0) {
        f->close();
        return false;
    }

    // write block0 to the file
    if (f->write(0, BLOCKSIZE, &block0) != MAPD_SUCCESS) {
        f->close();
        return false;
    }
    
    // write block1 to the file
    if (f->append(BLOCKSIZE, &block1) != MAPD_SUCCESS) {
        f->close();
        return false;
    }   

    // Check that the file is the correct size
    printf("fileSize=%lu\n", f->fileSize());
    if (f->fileSize() != 2*BLOCKSIZE) {
        f->close();
        return false;
    }

    return (f->close() == MAPD_SUCCESS);    
}

bool test_writeBlock() {
    return false;
}

bool test_append() {
    // create a file for testing
    if (f->open("test.db", true) != MAPD_SUCCESS)
        return false;

    // write block0 to the file
    if (f->append(BLOCKSIZE, &block0) != MAPD_SUCCESS) {
        f->close();
        return false;
    }
    
    // write block1 to the file
    if (f->append(BLOCKSIZE, &block1) != MAPD_SUCCESS) {
        f->close();
        return false;
    }   

    // Check that the file is the correct size
    if (f->fileSize() != 2*BLOCKSIZE) {
        f->close();
        return false;
    }

    return (f->close() == MAPD_SUCCESS);    
}

bool test_appendBlock() {
    return false;
}
