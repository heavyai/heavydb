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

using namespace Testing;

// unit test function prototypes
bool test_open_close();
bool test_create_read_write_append();
bool test_create_read_write_append_block();

#define BLOCKSIZE 10
#define FILEID 0

char block0[BLOCKSIZE];
char block1[BLOCKSIZE];

int main() {

    // populate block arrays used for testing
    for (int i = 0; i < BLOCKSIZE; i++) {
        block0[i] = i;
        block1[i] = BLOCKSIZE - i;
    }

    // call unit tests
    test_open_close() ? 
        PPASS("open_close()") : PFAIL("open_close()"); 
    test_create_read_write_append() ? 
          PPASS("create_read_write_append()") : PFAIL("create_read_write_append()"); 
    test_create_read_write_append_block() ? 
          PPASS("create_read_write_append_block()") : PFAIL("create_read_write_append_block()"); 
    
    printTestSummary();
    
    return 0;
}

bool test_open_close() {
    mapd_err_t err;
    FILE *f;
    
    f = File::create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;
    File::close(f);
    
    f = File::open(FILEID, NULL);
    if (!f) return false;
    return (File::close(f) == MAPD_SUCCESS);
}

bool test_create_read_write_append() {
    int i, j;
    char buf[BLOCKSIZE];
    size_t sz;
    mapd_err_t err;
    FILE *f;
    
    // create a file for testing
    f = File::create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;
    File::close(f);
    
    // open the file for testing
    f = File::open(FILEID, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;

    // write block0 to the file
    sz = File::write(f, 0, BLOCKSIZE, &block0, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        File::close(f);
        return false;
    }
    
    // append block1 to the file
    sz = File::append(f, BLOCKSIZE, &block1, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        File::close(f);
        return false;
    }

    // read block0 from the file and verify its contents
    sz = File::read(f, 0, BLOCKSIZE, &buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        File::close(f);
        return false;
    }
    for (i = 0; i < BLOCKSIZE; ++i) {
        if (block0[i] != buf[i]) {
            File::close(f);
            return false;
        }
    }

    // read block1 from the file and verify its contents
    sz = File::read(f, BLOCKSIZE, BLOCKSIZE, &buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        File::close(f);
        return false;
    }
    for (i = 0, j = BLOCKSIZE; i < BLOCKSIZE; ++i, ++j) {
        if (block1[i] != buf[i]) {
            File::close(f);
            return false;
        }
    }

    return (File::close(f) == MAPD_SUCCESS);
}

bool test_create_read_write_append_block() {
    int i, j;
    char buf[BLOCKSIZE];
    size_t sz;
    mapd_err_t err;
    FILE *f;
    
    // create a file for testing
    f = File::create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;
    File::close(f);
    
    // open the file for testing
    f = File::open(FILEID, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;

    // write block0 to the file
    sz = File::writeBlock(f, BLOCKSIZE, 0, &block0, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        File::close(f);
        return false;
    }
    
    // append block1 to the file
    sz = File::appendBlock(f, BLOCKSIZE, &block1, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        File::close(f);
        return false;
    }

    // read block0 from the file and verify its contents
    sz = File::readBlock(f, BLOCKSIZE, 0, &buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        File::close(f);
        return false;
    }
    for (i = 0; i < BLOCKSIZE; ++i) {
        if (block0[i] != buf[i]) {
            File::close(f);
            return false;
        }
    }

    // read block1 from the file and verify its contents
    sz = File::readBlock(f, BLOCKSIZE, 1, &buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        File::close(f);
        return false;
    }
    for (i = 0, j = BLOCKSIZE; i < BLOCKSIZE; ++i, ++j) {
        if (block1[i] != buf[i]) {
            File::close(f);
            return false;
        }
    }

    return (File::close(f) == MAPD_SUCCESS);
}

