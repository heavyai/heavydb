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

File *f = new File();

int main() {
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
    return (f->open("test.db", true));
}

bool test_close() {
    f->open("test.db", true);
    return (f->close() != MAPD_SUCCESS);
}

bool test_read() {
    f->open("test.db", true);
    return false;
}

bool test_readBlock() {
    return false;
}

bool test_write() {
    return false;
}

bool test_writeBlock() {
    return false;
}

bool test_append() {
    return false;
}

bool test_appendBlock() {
    return false;
}
