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
#include "../../Shared/macros.h"

using namespace Testing;
using namespace File_Namespace;

// unit test function prototypes
void test_Block();
void test_MultiBlock();
void test_push();
void test_pop();
void test_current();

int main() {

    // call unit tests
    test_Block();
    test_MultiBlock();
    test_push();
    test_pop();
    test_current();

    printTestSummary();
    
    return 0;
}

// Test Block's constructor
void test_Block() {
    const int fileId = 0;
    const mapd_size_t begin = 0;
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
    (mb.epoch.size() == 0) ? PPASS("MultiBlock epoch is empty") : PFAIL("MultiBlock epoch is not empty");
}

void test_push() {
    bool err;
    const mapd_size_t blockSize = 32;
    const int fileId = 0;
    int epoch = 0;
    const int N = 10;
    MultiBlock mb(fileId, blockSize);

    // create some Block objects
    Block* blk[N];
    for (int i = 0; i < N; ++i)
        blk[i] = new Block(i, (mapd_size_t)(i * blockSize));

    // push block versions into MultiBlock
    for (int i = 0; i < N; ++i)
        mb.push(blk[i], i);

    // verify the blocks were inserted successfully (mb.version)
    err = false;
    for (int i = 0; i < N; ++i) {
        if (mb.version[i] != blk[i]) {
            err = true;
            break;
        }
    }
    if (err)
        PFAIL("push() - incorrect block pointer stored in version vector");
    else
        PPASS("push() - version vector contains the inserted blocks");        

    // verify the blocks were inserted successfully (mb.epoch)
    err = false;
    for (int i = 0; i < N; ++i) {
        if (mb.epoch[i] != i) {
            err = true;
            break;
        }
    }
    if (err)
        PFAIL("push() - incorrect epoch stored in version vector");
    else
        PPASS("push() - epoch vector contains the correct values");               


    // free memory
    //for (int i = 0; i < N; ++i)
    //    delete blk[i];
}

void test_pop() {
    //printf("starting pop \n");
    bool err;
    const mapd_size_t blockSize = 32;
    const int fileId = 0;
    int epoch = 0;
    const int N = 10;
    MultiBlock mb(fileId, blockSize);

    // create some Block objects
    Block* blk[N];
    for (int i = 0; i < N; ++i)
        blk[i] = new Block(i, (mapd_size_t)(i * blockSize));

    // push block versions into MultiBlock. This relies on test_push() passing
    for (int i = 0; i < N; ++i)
        mb.push(blk[i], i);

    // let's get poppin'

    /* --------------------- epoch testing ------------------------*/
    err = false;
    for (int i = 0; i < N - 1; ++i) {
        mb.pop();

        // Has the epoch of the first block changed? Is the size appropriate?
        if ((mb.epoch.front() != i + 1) && (mb.epoch.size() != N - i - 1)) {
            err = true;
            break;
        }
    }

    // multiBlock now has a single block; check to see if it's empty after popping
    mb.pop();
    if (mb.epoch.size() != 0) err = true;

    if (err) {
        PFAIL("pop() - epoch vector not updated after pop");

    }
    else
        PPASS("pop() - epoch vector updated after pop");   

    //refill the blocks: create some Block objects
    for (int i = 0; i < N; ++i)
        blk[i] = new Block(i, (mapd_size_t)(i * blockSize));

    // push block versions into MultiBlock. This relies on test_push() passing
    for (int i = 0; i < N; ++i)
        mb.push(blk[i], i);

    /* --------------------- version testing ------------------------*/
    err = false;
    for (int i = 1; i < N; ++i) {
        mb.pop();

        // Has the epoch of the first block changed? Is the size appropriate?
        if ((mb.version.front() != blk[0]) && (mb.epoch.size() != N - i)) {
            err = true;
            break;
        }
    }

    // multiBlock now has a single block; check to see if it's empty after popping
    mb.pop();
    if (mb.version.size() != 0) err = true;

    if (err) {
        PFAIL("pop() - version vector not updated after pop");

    }
    else
        PPASS("pop() - version vector updated after pop");      

    // free memory
    //for (int i = 0; i < N; ++i)
    //    delete blk[i];
}

void test_current() {
    bool err;
    const mapd_size_t blockSize = 32;
    const int fileId = 0;
    int epoch = 0;
    const int N = 10;
    MultiBlock mb(fileId, blockSize);

    // create some Block objects
    Block* blk[N];
    for (int i = 0; i < N; ++i)
        blk[i] = new Block(i, (mapd_size_t)(i * blockSize));

    /* --------------------- version testing ------------------------*/

    // first, test to see if the current block is updated with each push
    err = false;
    for (int i = 0; i < N; ++i) {
        mb.push(blk[i], i);
        if (&mb.current() != blk[i]) {
            err = true;
            break; 
        }
    }
    if (err) {
        PFAIL("current() - current version not updated after push");
    }
    else
        PPASS("current() - current version update after push");  

    // printf("%lu\n", mb.version.size());
    // second, test to see if the current block is updated with each pop
    err = false;
    for (int i = 0; i < N - 1; ++i) {
        mb.version.pop_back();

        if (&mb.current() != blk[mb.version.size() - 1]) {
            err = true;
            break;
        }
    }

    // clear the multiBlock
    mb.pop();

    if (err) {
        PFAIL("current() - current version not updated after pop");
    }
    else
        PPASS("current() - current version update after pop");  

    // create some Block objects
    for (int i = 0; i < N; ++i)
        blk[i] = new Block(i, (mapd_size_t)(i * blockSize));
    /* --------------------- epoch testing ------------------------*/
    // first, test to see if the current epoch is updated with each push
    int queryEpoch;
    err = false;
    for (int i = 0; i < N; ++i) {
        mb.push(blk[i], i);

        mb.current(&queryEpoch);
        if (queryEpoch != i) {
            err = true;
            break; 
        }
    }
    if (err) {
        PFAIL("current() - current epoch not updated after push");
    }
    else
        PPASS("current() - current epoch update after push");  
    
    // second, test to see if the current block is updated with each pop
    err = false;
    for (int i = 0; i < N - 1; ++i) {
        mb.version.pop_back();

        if (&mb.current() != blk[mb.version.size() - 1]) {
            err = true;
            break;
        }
    }

    // clear the multiBlock
    mb.pop();

    if (err) {
        PFAIL("current() - current version not updated after pop");
    }
    else
        PPASS("current() - current version update after pop");  
}