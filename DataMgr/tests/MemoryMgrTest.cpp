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

TEST_F (MemoryMgrTest, pushWriteAndRead) {
    ChunkKey key;
    for (int i = 0; i < 3; ++i) {
        key.push_back(i);
    }
    AbstractBuffer * gpuBuffer = memoryMgr -> createChunk(GPU_LEVEL,key);
    const int numInts = 10000;
    int * data1 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    gpuBuffer -> append((mapd_addr_t)data1,numInts*sizeof(int),Memory_Namespace::CPU_BUFFER);
    // will put gpu chunk to cpu chunk, and then cpu chunk to file chunk
    memoryMgr -> checkpoint();
    // will read File chunk into Cpu chunk
    AbstractBuffer * cpuBuffer = memoryMgr -> getChunk(CPU_LEVEL,key);
    int *data2 = new int [numInts];
    cpuBuffer -> read((mapd_addr_t)data2,numInts*sizeof(int),Memory_Namespace::CPU_BUFFER,0);
    for (size_t i = 0; i < numInts; ++i) {
        EXPECT_EQ(data1[i],data2[i]);
    }

    delete [] data1;
    delete [] data2;
}

TEST_F (MemoryMgrTest, deleteChunk) {
    ChunkKey key;
    for (int i = 0; i < 3; ++i) {
        key.push_back(i);
    }
    AbstractBuffer * cpuBuffer = memoryMgr -> createChunk(CPU_LEVEL,key);
    const int numInts = 10000;
    int * data1 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    cpuBuffer -> append((mapd_addr_t)data1,numInts*sizeof(int),Memory_Namespace::CPU_BUFFER);
    memoryMgr -> checkpoint();
    EXPECT_NO_THROW(memoryMgr -> getChunk(GPU_LEVEL,key));
    EXPECT_NO_THROW(memoryMgr -> getChunk(CPU_LEVEL,key));
    EXPECT_NO_THROW(memoryMgr -> getChunk(DISK_LEVEL,key));
    memoryMgr -> deleteChunk(key);
    EXPECT_THROW(memoryMgr -> getChunk(GPU_LEVEL,key), std::runtime_error);
    EXPECT_THROW(memoryMgr -> getChunk(CPU_LEVEL,key), std::runtime_error);
    EXPECT_THROW(memoryMgr -> getChunk(DISK_LEVEL,key), std::runtime_error);
    delete [] data1;
}












