#include "gtest/gtest.h"
#include "../MemoryMgr.h"

#include <boost/filesystem.hpp>

#include <iostream>

using namespace Memory_Namespace;
using namespace std;

#define MEMORYMGR_UNIT_TESTING

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class MemoryMgrTest : public ::testing::Test {

    protected:
        virtual void SetUp() {
            deleteData("data");
            memoryMgr = new MemoryMgr(2,"data");
        }

        virtual void TearDown() {
            delete memoryMgr;
        }

        void deleteData(const std::string &dirName) {
            boost::filesystem::remove_all(dirName);
        }

        MemoryMgr *memoryMgr;
};

TEST_F (MemoryMgrTest, createChunk) {
    ChunkKey key;
    for (int i = 0; i < 3; ++i) {
        key.push_back(i);
    }
    AbstractBuffer * buffer = memoryMgr -> createChunk(GPU_LEVEL,key);
    const int numInts = 10000;
    int * data1 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    buffer -> append((mapd_addr_t)data1,numInts*sizeof(int),Memory_Namespace::CPU_BUFFER);
    memoryMgr -> checkpoint();
    memoryMgr -> checkpoint();

    //ASSERT_NEQ(memoryMgr, 0);
}




