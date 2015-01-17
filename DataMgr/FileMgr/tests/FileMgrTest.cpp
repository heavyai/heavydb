#include "../FileMgr.h"

#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>

#include <boost/filesystem.hpp>

#include "gtest/gtest.h"
#include <boost/timer/timer.hpp>

using namespace File_Namespace;
using namespace std;

void deleteData(const std::string &dirName) {
    boost::filesystem::remove_all(dirName);
}

void writeToBuffer(AbstractBuffer *buffer, const size_t numInts) {
    int * data = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data[i] = i;
    }
    buffer -> write((mapd_addr_t)data,numInts*sizeof(int),CPU_BUFFER,0);
    delete [] data;
}

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(FileMgr, getFreePages)
{
    deleteData("data");
    FileMgr fm("data");
    std::vector<Page> freePages;
    mapd_size_t numPages = 2048;
    mapd_size_t pageSize = 4096;
    
    EXPECT_EQ(freePages.size(), 0);
    fm.requestFreePages(numPages, pageSize, freePages);
    EXPECT_EQ(freePages.size(), numPages);
    
}

TEST(FileMgr, getFreePage)
{
    deleteData("data");
    FileMgr fm("data");
    mapd_size_t pageSize = 1024796;
    Page page = fm.requestFreePage(pageSize);
    EXPECT_EQ(page.isValid(),true);
}

TEST(FileMgr, createChunk) {
    deleteData("data");
    FileMgr fm("data");
    ChunkKey chunkKey = {2,3,4,5};
    mapd_size_t pageSize = 4096;
    AbstractBuffer * chunk1 = fm.createChunk(chunkKey,pageSize);
    AbstractBuffer * chunk2 = fm.getChunk(chunkKey);
    EXPECT_EQ(chunk1,chunk2);
    // Creating same chunk again should fail
    try {
        fm.createChunk(chunkKey,pageSize);
        EXPECT_TRUE(1==2) << "Created two chunks with same chunk key";
    }
    catch (std::runtime_error &error) {
        string expectedErrorString ("Chunk already exists.");
        EXPECT_EQ(error.what(),expectedErrorString);
    }
}

TEST(FileMgr, deleteChunk) {
    deleteData("data");
    ChunkKey chunkKey1 = {2,3,4,5};
    ChunkKey chunkKey2 = {2,4,4,5};
    {
        FileMgr fm("data");
        mapd_size_t pageSize = 4096;
        fm.createChunk(chunkKey1,pageSize);
        AbstractBuffer *chunk = fm.getChunk(chunkKey1);
        writeToBuffer(chunk,4096);
        // Test 1: Try to delete chunk
        try {
            fm.deleteChunk(chunkKey1); // should succeed
            EXPECT_TRUE(1==1);
        }
        catch (std::runtime_error &error) {
            EXPECT_TRUE(1==2) << "Could not delete a chunk that does exist";
        }

        // Test 2: Try to get chunk after its deleted
        try {
            AbstractBuffer *chunk1 = fm.getChunk(chunkKey1);
            EXPECT_TRUE(1==2) << "getChunk succeeded on a chunk that was deleted";
        }
        catch (std::runtime_error &error) {
            string expectedErrorString ("Chunk does not exist.");
            EXPECT_EQ(error.what(),expectedErrorString);
        }

        // Test 3: Try to delete chunk that was never created
        try {
            fm.deleteChunk(chunkKey2);
            EXPECT_TRUE(1==2) << "Deleted chunk that was never created";
        }
        catch (std::runtime_error &error) {
            string expectedErrorString ("Chunk does not exist.");
            EXPECT_EQ(error.what(),expectedErrorString);
        }

        // Test 4: Try to delete chunk that had previously been deleted
        try {
            fm.deleteChunk(chunkKey1);
            EXPECT_TRUE(1==2) << "Deleted chunk that had already been deleted";
        }
        catch (std::runtime_error &error) {
            string expectedErrorString ("Chunk does not exist.");
            EXPECT_EQ(error.what(),expectedErrorString);
        }
        fm.checkpoint();
    }
    // Test 5: Destroy FileMgr and reinstanciate it to make sure it has no
    // trace of chunk with key chunkKey1
    {
        FileMgr fm("data");
        try {
            AbstractBuffer *chunk1 = fm.getChunk(chunkKey1);
            EXPECT_TRUE(1==2) << "getChunk succeeded on a chunk that was deleted after FileMgr reinstanciation";
        }
        catch (std::runtime_error &error) {
            string expectedErrorString ("Chunk does not exist.");
            EXPECT_EQ(error.what(),expectedErrorString);
        }
    }
}

TEST(FileMgr, writeReadChunk) {
    deleteData("data");
    ChunkKey chunkKey1 = {1,2,3,4};
    ChunkKey chunkKey2 = {2,3,4,5};
    mapd_size_t pageSize = 1024796;
    //mapd_size_t pageSize = 8192;
    //mapd_size_t pageSize = 4096000;
    FileMgr fm("data");
    fm.createChunk(chunkKey1,pageSize);
    fm.createChunk(chunkKey2,pageSize);
    const boost::timer::nanosecond_type oneSecond(1000000000LL);
    size_t numInts = 10000000;
    int * data1 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    AbstractBuffer *chunk1 = fm.getChunk(chunkKey1);
    {
        boost::timer::cpu_timer cpuTimer;
        chunk1 -> write((mapd_addr_t)data1,numInts*sizeof(int),CPU_BUFFER,0);
        cout << "Checkpoint 1" << endl;
        fm.checkpoint();
        double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond;
        double bandwidth = numInts * sizeof(int) / elapsedTime / 1000000000.0;
        cout << "Write Bandwidth with checkpoint: " << bandwidth << " GB/sec" << endl;
    }
    AbstractBuffer *chunk2 = fm.getChunk(chunkKey2);
    {
        boost::timer::cpu_timer cpuTimer;
        chunk1 -> write((mapd_addr_t)data1,numInts*sizeof(int),CPU_BUFFER,0);
        double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond;
        double bandwidth = numInts * sizeof(int) / elapsedTime / 1000000000.0;
        cout << "Write Bandwidth without checkpoint: " << bandwidth << " GB/sec" << endl;
    }

    int * data2 = new int [numInts];
    {
        boost::timer::cpu_timer cpuTimer;
        chunk1 -> read((mapd_addr_t)data2,numInts*sizeof(int),CPU_BUFFER,0);
        double elapsedTime = double(cpuTimer.elapsed().wall) / oneSecond;
        double bandwidth = numInts * sizeof(int) / elapsedTime / 1000000000.0;
        cout << "Read Bandwidth: " << bandwidth << " GB/sec" << endl;
    }

    for (size_t i = 0; i < numInts; ++i) {
        EXPECT_EQ(data1[i],data2[i]);
    }

    delete [] data1;
    delete [] data2;
}

TEST(FileMgr, persistence) {
    deleteData("data");
    ChunkKey chunkKey1 = {1,2,3,4};
    mapd_size_t pageSize = 1024796;
    size_t numInts = 1000000;
    int * data1 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    {
        FileMgr fm("data");
        fm.createChunk(chunkKey1,pageSize);
        AbstractBuffer *chunk1 = fm.getChunk(chunkKey1);
        chunk1 -> write((mapd_addr_t)data1,numInts*sizeof(int),CPU_BUFFER,0);
        fm.checkpoint();
        // should checkpoint here 
    }
    FileMgr fm("data");
    AbstractBuffer * chunk1 = fm.getChunk(chunkKey1);
    EXPECT_EQ(chunk1 -> size(),numInts * sizeof(int));
    delete [] data1;
}

TEST(FileMgr, epochPersistence) {
    deleteData("data");
    ChunkKey chunkKey1 = {1,2,3,4};
    mapd_size_t pageSize = 1024796;
    size_t numInts = 100000;
    int * data1 = new int [numInts];
    for (size_t i = 0; i < numInts; ++i) {
        data1[i] = i;
    }
    {
        FileMgr fm("data");
        fm.createChunk(chunkKey1,pageSize);
        AbstractBuffer *chunk1 = fm.getChunk(chunkKey1);
        chunk1 -> append((mapd_addr_t)data1,numInts*sizeof(int),CPU_BUFFER);
        cout << "After checkpoint 1 for epoch Persistence" << endl;
        fm.checkpoint(); // checkpoint 1
        chunk1 -> append((mapd_addr_t)data1,numInts*sizeof(int),CPU_BUFFER);
        fm.checkpoint(); // checkpoint 2
        chunk1 -> append((mapd_addr_t)data1,numInts*sizeof(int),CPU_BUFFER);
        fm.checkpoint(); // checkpoint 3
        chunk1 -> append((mapd_addr_t)data1,numInts*sizeof(int),CPU_BUFFER);
        fm.checkpoint(); // checkpoint 4
    }
    cout << "After checkpoints for epoch Persistence" << endl;



    {
        cout << "Test 1" << endl;
        FileMgr fm("data");
        AbstractBuffer * chunk1 = fm.getChunk(chunkKey1);
        EXPECT_EQ(chunk1 -> size(),4*numInts * sizeof(int));
    }
    {
        cout << "Test 2" << endl;
        FileMgr fm("data", 1024796, 3);
        AbstractBuffer * chunk1 = fm.getChunk(chunkKey1);
        EXPECT_EQ(chunk1 -> size(),3*numInts * sizeof(int));
    }
    {
        FileMgr fm("data", 1024796, 2);
        AbstractBuffer * chunk1 = fm.getChunk(chunkKey1);
        EXPECT_EQ(chunk1 -> size(),2*numInts * sizeof(int));
    }
    {
        FileMgr fm("data", 1024796, 1);
        AbstractBuffer * chunk1 = fm.getChunk(chunkKey1);
        EXPECT_EQ(chunk1 -> size(),1*numInts * sizeof(int));
    }
}









