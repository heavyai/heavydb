#include "gtest/gtest.h"
#include "../DataMgr.h"

#include <boost/filesystem.hpp>

#include <iostream>

using namespace std;

#define DATAMGR_UNIT_TESTING


GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace Data_Namespace {

    class DataMgrTest : public ::testing::Test {

        protected:
            virtual void SetUp() {
                deleteData("data");
                dataMgr = new DataMgr(2,"data");
            }

            virtual void TearDown() {
                delete dataMgr;
            }

            void deleteData(const std::string &dirName) {
                boost::filesystem::remove_all(dirName);
            }

            DataMgr *dataMgr;
    };

    TEST_F (DataMgrTest, pushWriteAndRead) {
        ChunkKey key;
        for (int i = 0; i < 3; ++i) {
            key.push_back(i);
        }
        AbstractBuffer * gpuBuffer = dataMgr -> createChunk(GPU_LEVEL,key);
        const int numInts = 10000;
        int * data1 = new int [numInts];
        for (size_t i = 0; i < numInts; ++i) {
            data1[i] = i;
        }
        gpuBuffer -> append((mapd_addr_t)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        // will put gpu chunk to cpu chunk, and then cpu chunk to file chunk
        dataMgr -> checkpoint();
        // will read File chunk into Cpu chunk
        AbstractBuffer * cpuBuffer = dataMgr -> getChunk(CPU_LEVEL,key);
        int *data2 = new int [numInts];
        cpuBuffer -> read((mapd_addr_t)data2,numInts*sizeof(int),Data_Namespace::CPU_BUFFER,0);
        for (size_t i = 0; i < numInts; ++i) {
            EXPECT_EQ(data1[i],data2[i]);
        }

        delete [] data1;
        delete [] data2;
    }

    TEST_F (DataMgrTest, deleteChunk) {
        ChunkKey key;
        for (int i = 0; i < 3; ++i) {
            key.push_back(i);
        }
        AbstractBuffer * cpuChunk = dataMgr -> createChunk(CPU_LEVEL,key);
        const int numInts = 10000;
        int * data1 = new int [numInts];
        for (size_t i = 0; i < numInts; ++i) {
            data1[i] = i;
        }
        cpuChunk -> append((mapd_addr_t)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        dataMgr -> checkpoint();
        EXPECT_NO_THROW(dataMgr -> getChunk(GPU_LEVEL,key));
        EXPECT_NO_THROW(dataMgr -> getChunk(CPU_LEVEL,key));
        EXPECT_NO_THROW(dataMgr -> getChunk(DISK_LEVEL,key));
        dataMgr -> deleteChunk(key);
        EXPECT_THROW(dataMgr -> getChunk(GPU_LEVEL,key), std::runtime_error);
        EXPECT_THROW(dataMgr -> getChunk(CPU_LEVEL,key), std::runtime_error);
        EXPECT_THROW(dataMgr -> getChunk(DISK_LEVEL,key), std::runtime_error);
        delete [] data1;
    }

    TEST_F (DataMgrTest, buffer) {
        AbstractBuffer *cpuBuffer = dataMgr -> createBuffer(CPU_LEVEL,0,4096);
        const int numInts = 10000;
        int * data1 = new int [numInts];
        for (size_t i = 0; i < numInts; ++i) {
            data1[i] = i;
        }
        cpuBuffer -> append((mapd_addr_t)data1,numInts*sizeof(int),Data_Namespace::CPU_BUFFER);
        //Checkpoint should not flush a buffer - only real chunks 
        dataMgr -> checkpoint();

        EXPECT_EQ(dataMgr -> bufferMgrs_[1][0] -> getNumChunks(),1);
        EXPECT_EQ(dataMgr -> bufferMgrs_[0][0] -> getNumChunks(),0);
    }
}















