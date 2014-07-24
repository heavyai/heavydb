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
void test_open_close();
void test_create_read_write_append();
void test_create_read_write_append_block();
void test_removeFile();

#define BLOCKSIZE 10
#define FILEID 0
#define MAPD_FILE_NAME_1 "0.mapd"
#define MAPD_FILE_NAME_2 "1.mapd"
#define MAPD_FILE_NAME_3 "2.mapd"

mapd_byte_t block0[BLOCKSIZE];
mapd_byte_t block1[BLOCKSIZE];

int main() {

    // populate block arrays used for testing
    for (int i = 0; i < BLOCKSIZE; i++) {
        block0[i] = i;
        block1[i] = BLOCKSIZE - i;
    }

    // call unit tests
    test_open_close();  
    test_create_read_write_append();
    test_create_read_write_append_block();
    test_removeFile();

    printTestSummary();
    
    return 0;
}

void test_open_close() {
    mapd_err_t err;
    FILE *f;
    
    f = create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        PFAIL("open_close() - unsuccessful creation");
    else PPASS("open_close() - successful creation");
    close(f);
    
    f = open(FILEID, NULL);
    if (!f) PFAIL("open_close() - unsuccessful open");
    PPASS("open_close() - successful open");
    
    if (close(f) == MAPD_SUCCESS) PPASS("open_close() - successful close");
    else PFAIL("open_close() - unsuccessful close");
}

void test_create_read_write_append() {
    int i, j;
    mapd_byte_t buf[BLOCKSIZE];
    size_t sz;
    mapd_err_t err;
    FILE *f;
    bool loopError;
    
    // create a file for testing
    f = create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS) PFAIL("test_create_read_write_append() - unsuccessful create");
    else PPASS("test_create_read_write_append() - successful create");
    close(f);
    
    // open the file for testing
    f = open(FILEID, &err);
    if (!f || err != MAPD_SUCCESS)
        PFAIL("test_create_read_write_append()- unsuccessful open");
    else PPASS("test_create_read_write_append() - successful open");

    // write block0 to the file
    sz = write(f, 0, BLOCKSIZE, (mapd_addr_t)block0, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        //close(f);
        PFAIL("test_create_read_write_append() - unsuccessful write (block0)");
    }
    else PPASS("test_create_read_write_append() - successful write (block0)");
    
    
    // append block1 to the file
    sz = append(f, BLOCKSIZE, (mapd_addr_t)block1, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        //close(f);
        PFAIL("test_create_read_write_append() - unsuccessful append (block1)");
    }
    else PPASS("test_create_read_write_append() - unsuccessful append (block1)");

    // read block0 from the file and verify its contents
    sz = read(f, 0, BLOCKSIZE, (mapd_addr_t)buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        //close(f);
        PFAIL("test_create_read_write_append() - unsuccessful read (block0)");
    }
    else PPASS("test_create_read_write_append() - successful read (block0)");

    loopError = false;
    for (i = 0; i < BLOCKSIZE; ++i) {
        if (block0[i] != buf[i]) {
            loopError = true;
        }
    }
    if (loopError) PFAIL("test_create_read_write_append() - bytes in block0 mismatch read bytes");
    else PPASS("test_create_read_write_append() - all bytes in block0 match read bytes");

    // read block1 from the file and verify its contents
    sz = read(f, BLOCKSIZE, BLOCKSIZE, (mapd_addr_t)buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        PFAIL("test_create_read_write_append() - unsuccessful read (block0)");
    }
    else PPASS("test_create_read_write_append() - successful read (block0)");

    loopError = false;
    for (i = 0, j = BLOCKSIZE; i < BLOCKSIZE; ++i, ++j) {
        if (block1[i] != buf[i]) {
            loopError = true;
        }
    }
    if (loopError) PFAIL("test_create_read_write_append() - bytes in block0 mismatch read bytes");
    else PPASS("test_create_read_write_append() - all bytes in block0 match read bytes");

    if (close(f) == MAPD_SUCCESS) PPASS("test_create_read_write_append() - successful close");
    else PFAIL("test_create_read_write_append() - unsuccessful close");
}

void test_create_read_write_append_block() {
    int i, j;
    mapd_byte_t buf[BLOCKSIZE];
    size_t sz;
    mapd_err_t err;
    FILE *f;
    bool loopError;
    
    // create a file for testing
    f = create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        PFAIL("test_create_read_write_append_block() - unsuccessful create");
    else PPASS("test_create_read_write_append_block() - successful create");
    close(f);
    
    // open the file for testing
    f = open(FILEID, &err);
    if (!f || err != MAPD_SUCCESS)
        PFAIL("test_create_read_write_append_block() - unsuccessful open");
    PPASS("test_create_read_write_append_block() - successful open");

    // write block0 to the file
    sz = writeBlock(f, BLOCKSIZE, 0, (mapd_addr_t)block0, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        PFAIL("test_create_read_write_append_block() - unsuccessful write (block0)");
    }
    else PPASS("test_create_read_write_append_block() - successful write (block0)");

    // append block1 to the file
    sz = appendBlock(f, BLOCKSIZE, (mapd_addr_t)block1, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        PFAIL("test_create_read_write_append_block() - unsuccessful append (block1)");
    }
    else PPASS("test_create_read_write_append_block() - successful append (block1)");

    // read block0 from the file and verify its contents
    sz = readBlock(f, BLOCKSIZE, 0, (mapd_addr_t)buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        PFAIL("test_create_read_write_append_block() - unsuccessful read (block0)");
    }
    else PPASS("test_create_read_write_append_block() - successful read (block0");

    loopError = false;
    for (i = 0; i < BLOCKSIZE; ++i) {
        if (block0[i] != buf[i]) {
            loopError = true;
        }
    }
    if (loopError) PFAIL("test_create_read_write_append_block() - bytes in block0 mismatch read bytes");
    else PPASS("test_create_read_write_append_block() - all bytes in block0 match read bytes");

    // read block1 from the file and verify its contents
    sz = readBlock(f, BLOCKSIZE, 1, (mapd_addr_t)buf, &err);
    if (err != MAPD_SUCCESS || sz != BLOCKSIZE) {
        PFAIL("test_create_read_write_append_block() - unsuccessful read (block1)");
    }
    else PPASS("test_create_read_write_append_block() - unsuccessful read (block1)");

    loopError = false;
    for (i = 0, j = BLOCKSIZE; i < BLOCKSIZE; ++i, ++j) {
        if (block1[i] != buf[i]) {
            loopError = true;
        }
    }
    if (loopError) PFAIL("test_create_read_write_append_block() - bytes in block1 mismatch read bytes");
    else PPASS("test_create_read_write_append_block() - all bytes in block1 match read bytes");

    if (close(f) == MAPD_SUCCESS) PPASS("test_create_read_write_append_block() - successful close");
    else PFAIL("test_create_read_write_append_block() - unsuccessful close");
}

void test_removeFile() {
    mapd_byte_t buf[BLOCKSIZE];
    size_t sz;
    mapd_err_t err;
    FILE *f;
    const std::string basePath = "";

    // create a file for testing
    f = create(FILEID, BLOCKSIZE, 1, &err);
    if (!f || err != MAPD_SUCCESS)
        PFAIL("test_removeFile() - unsuccessful create");
    else PPASS("test_removeFile() - successful create");

    close(f);
    const std::string filename1 = MAPD_FILE_NAME_1;

    //std::cout << filename << std::endl;

    if (removeFile(basePath, filename1) != MAPD_SUCCESS)
        PFAIL("test_removeFile() - unsuccessful removeFile (1)");
    else PPASS("test_removeFile() - successful removeFile (1)");

    f = create(FILEID + 1, BLOCKSIZE, 1, &err);
    close(f);
    open(FILEID + 1, &err);
    close(f);

    const std::string filename2 = MAPD_FILE_NAME_2;
    if (removeFile(basePath, filename2) != MAPD_SUCCESS)
        PFAIL("test_removeFile() - unsuccessful removeFile (2)");
    else PPASS("test_removeFile() - successful removeFile (2)");

    f = create(FILEID + 2, BLOCKSIZE, 1, &err);
    close(f);
    f = open(FILEID + 2, &err);
    sz = writeBlock(f, BLOCKSIZE, 0, (mapd_addr_t)block0, &err);
    close(f);
    
    const std::string filename3 = MAPD_FILE_NAME_3;
    if (removeFile(basePath, filename3) != MAPD_SUCCESS)
        PFAIL("test_removeFile() - unsuccessful removeFile (3)");
    else PPASS("test_removeFile() - successful removeFile (3)");
}