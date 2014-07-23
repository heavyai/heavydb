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
using namespace File_Namespace;

// unit test function prototypes
bool test_open_close();
bool test_create_read_write_append();
bool test_create_read_write_append_block();
bool test_erase();

#define BLOCKSIZE 10
#define FILEID 0
#define MAPD_FILE_NAME "0.mapd"

mapd_byte_t block0[BLOCKSIZE];
mapd_byte_t block1[BLOCKSIZE];

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
    test_erase() ? 
          PPASS("erase()") : PFAIL("erase()"); 
   
    printTestSummary();
    
    return 0;
}

bool test_open_close() {
    mapd_err_t err;
    FILE *f;
    
    f = create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;
    close(f);
    
    f = open(FILEID, NULL);
    if (!f) return false;
    return (close(f) == MAPD_SUCCESS);
}

bool test_create_read_write_append() {
    int i, j;
    mapd_byte_t buf[BLOCKSIZE];
    size_t sz;
    mapd_err_t err;
    FILE *f;
    
    // create a file for testing
    f = create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;
    close(f);
    
    // open the file for testing
    f = open(FILEID, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;

    // write block0 to the file
    sz = write(f, 0, BLOCKSIZE, (mapd_addr_t)block0, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        close(f);
        return false;
    }
    
    // append block1 to the file
    sz = append(f, BLOCKSIZE, (mapd_addr_t)block1, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        close(f);
        return false;
    }

    // read block0 from the file and verify its contents
    sz = read(f, 0, BLOCKSIZE, (mapd_addr_t)buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        close(f);
        return false;
    }
    for (i = 0; i < BLOCKSIZE; ++i) {
        if (block0[i] != buf[i]) {
            close(f);
            return false;
        }
    }

    // read block1 from the file and verify its contents
    sz = read(f, BLOCKSIZE, BLOCKSIZE, (mapd_addr_t)buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        close(f);
        return false;
    }
    for (i = 0, j = BLOCKSIZE; i < BLOCKSIZE; ++i, ++j) {
        if (block1[i] != buf[i]) {
            close(f);
            return false;
        }
    }

    return (close(f) == MAPD_SUCCESS);
}

bool test_create_read_write_append_block() {
    int i, j;
    mapd_byte_t buf[BLOCKSIZE];
    size_t sz;
    mapd_err_t err;
    FILE *f;
    
    // create a file for testing
    f = create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;
    close(f);
    
    // open the file for testing
    f = open(FILEID, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;

    // write block0 to the file
    sz = writeBlock(f, BLOCKSIZE, 0, (mapd_addr_t)block0, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        close(f);
        return false;
    }
    
    // append block1 to the file
    sz = appendBlock(f, BLOCKSIZE, (mapd_addr_t)block1, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        close(f);
        return false;
    }

    // read block0 from the file and verify its contents
    sz = readBlock(f, BLOCKSIZE, 0, (mapd_addr_t)buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        close(f);
        return false;
    }
    for (i = 0; i < BLOCKSIZE; ++i) {
        if (block0[i] != buf[i]) {
            close(f);
            return false;
        }
    }

    // read block1 from the file and verify its contents
    sz = readBlock(f, BLOCKSIZE, 1, (mapd_addr_t)buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        close(f);
        return false;
    }
    for (i = 0, j = BLOCKSIZE; i < BLOCKSIZE; ++i, ++j) {
        if (block1[i] != buf[i]) {
            close(f);
            return false;
        }
    }

    return (close(f) == MAPD_SUCCESS);
}

bool test_erase() {
    mapd_byte_t buf[BLOCKSIZE];
    size_t sz;
    mapd_err_t err;
    FILE *f;
    
    const std::string basePath = "";

    // create a file for testing
    f = create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        return false;
    close(f);
    const std::string filename = MAPD_FILE_NAME;

    //std::cout << filename << std::endl;

    if (erase(basePath, filename) != MAPD_SUCCESS)
        return false;

    f = create(FILEID, BLOCKSIZE, 1, &err);
    close(f);
    open(FILEID, &err);
    close(f);

    if (erase(basePath, filename) != MAPD_SUCCESS)
        return false;

    f = create(FILEID, BLOCKSIZE, 1, &err);
    close(f);
    f = open(FILEID, &err);
    sz = writeBlock(f, BLOCKSIZE, 0, (mapd_addr_t)block0, &err);
    close(f);
    
    if (erase(basePath, filename) != MAPD_SUCCESS)
        return false;
    
    return true;
}