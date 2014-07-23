/**
 * File:       blockTest.cpp
 * Author(s):  steve@map-d.com
 * 
 * This program is intended to test the interface specified in Block.h.
 * Note that Block.h contains inlined functions and does not have a 
 * corresponding .cpp file. 
 */
#include <cstdio>
#include <cstdlib>
#include "Block.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

using namespace Testing;
using namespace File_Namespace;

// unit test function prototypes
void test_Block();
void test_MultiBlock();
void test_pop();

int main() {

    // call unit tests
    test_Block();
    test_MultiBlock();
    test_pop();

    printTestSummary();
    
    return 0;
}

// Test Block's constructor
void test_Block() {
    const int fileId = 0;
    const mapd_addr_t begin = 0;
    Block blk(fileId, begin);

    // The member variables should be set correctly
    (blk.fileId == fileId) ? PPASS("Block fileId set correctly") : PFAIL("Block fileId not set correctly");
    (blk.begin == begin) ? PPASS("Block begin set correctly") : PFAIL("Block begin not set correctly");
    (blk.end == begin) ? PPASS("Block end set correctly") : PFAIL("Block end not set correctly");
}

// Test MultiBlock's constructor
void test_MultiBlock() {
    const mapd_size_t blockSize = 32;
    const int fileId = 0;
    int epoch = 0;
    MultiBlock mb(fileId, blockSize);

    // The member variables should be set correctly
    (mb.fileId == fileId) ? PPASS("MultiBlock fileId set correctly") : PFAIL("MultiBlock fileId not set correctly");
    (mb.blockSize == blockSize) ? PPASS("MultiBlock blockSize set correctly") : PFAIL("MultiBlock blockSize not set correctly");
    (mb.version.size() == 0) ? PPASS("MultiBlock version is empty") : PFAIL("MultiBlock version is not empty");
    (mb.version.size() == 0) ? PPASS("MultiBlock epoch is empty") : PFAIL("MultiBlock epoch is not empty");
}

void test_pop() {
    const mapd_size_t blockSize = 32;
    const int fileId = 0;
    int epoch = 0;
    const int N = 10;
    MultiBlock mb(fileId, blockSize);

    // create some Block objects
    Block *blk[N];
    for (int i = 0; i < N; ++i)
        blk[i] = new Block(i, (mapd_addr_t)(i * blockSize));

    // push block versions into MultiBlock
    for (int i = 0; i < N; ++i)
        mb.push(blk[i], i);

    // verify the blocks were inserted successfully
    bool err = false;
    for (int i = 0; i < N; ++i) {
        if (mb.version[i] != blk[i]) {
            err = true;
            break;
        }
    }
    if (err)
        PFAIL("push() - incorrect block pointer found");
    else
        PPASS("push() - version contains the inserted blocks");        

    // free memory
    for (int i = 0; i < N; ++i)
        delete blk[i];

}
