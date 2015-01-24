/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and related
 * data structures and types.
 */
#ifndef DATAMGR_MEMORY_BUFFER_BUFFERMGR_H
#define DATAMGR_MEMORY_BUFFER_BUFFERMGR_H

#include <iostream>
#include <map>
#include <list>
#include "../AbstractBuffer.h"
#include "../AbstractBufferMgr.h"
#include "BufferSeg.h"
#include <gtest/gtest_prod.h>
#include <mutex>

using namespace Data_Namespace;

namespace Buffer_Namespace {


    /**
     * @class   BufferMgr
     * @brief
     *
     * Note(s): Forbid Copying Idiom 4.1
     */


    class BufferMgr : public AbstractBufferMgr { // implements
        
    public:
        //FRIEND_TEST(BufferMgrTest, slabTest);
        //friend class BufferMgrTest_slabTest_Test;
        
        /// Constructs a BufferMgr object that allocates memSize bytes.
        //@todo change this to size_t
        //explicit BufferMgr(const size_t bufferSize, const size_t pageSize);
        BufferMgr(const size_t maxBufferSize, const size_t bufferAllocIncrement = 2147483648,  const size_t pageSize = 512, AbstractBufferMgr *parentMgr = 0);
        
        /// Destructor
        virtual ~BufferMgr();

        void clear();

        void printMap();
        void printSegs();
        void printSeg(BufferList::iterator &segIt);
        
        /// Creates a chunk with the specified key and page size.
        virtual AbstractBuffer * createChunk(const ChunkKey &key, const size_t pageSize = 0, const size_t initialSize = 0);
        
        /// Deletes the chunk with the specified key
        virtual void deleteChunk(const ChunkKey &key);
        
        /// Returns the a pointer to the chunk with the specified key.
        virtual AbstractBuffer* getChunk(const ChunkKey &key, const size_t numBytes = 0);
        
        /**
         * @brief Puts the contents of d into the Buffer with ChunkKey key.
         * @param key - Unique identifier for a Chunk.
         * @param d - An object representing the source data for the Chunk.
         * @return AbstractBuffer*
         */
        virtual void fetchChunk(const ChunkKey &key, AbstractBuffer *destBuffer, const size_t numBytes = 0);
        virtual AbstractBuffer* putChunk(const ChunkKey &key, AbstractBuffer *d, const size_t numBytes = 0);
        void checkpoint();

        // Buffer API
        virtual AbstractBuffer* createBuffer(const size_t numBytes = 0);
        virtual void deleteBuffer(AbstractBuffer *buffer);
        //virtual AbstractBuffer* putBuffer(AbstractBuffer *d);
        
        /// Returns the total number of bytes allocated.
        size_t size();
        size_t getNumChunks();

        BufferList::iterator reserveBuffer(BufferList::iterator & segIt, const size_t numBytes);
        virtual void getChunkMetadataVec(std::vector<std::pair<ChunkKey,ChunkMetadata> > &chunkMetadataVec);

       
    protected: 
        std::vector <int8_t *> slabs_;       /// vector of beginning memory addresses for each allocation of the buffer pool
        std::vector<BufferList> slabSegments_; // last list is for unsized segments
        size_t numPagesPerSlab_;

    private:
        BufferMgr(const BufferMgr&); // private copy constructor
        BufferMgr& operator=(const BufferMgr&); // private assignment
         void removeSegment(BufferList::iterator &segIt);
        BufferList::iterator findFreeBufferInSlab(const size_t slabNum, const size_t numPagesRequested);
        int getBufferId();
        virtual void addSlab(const size_t slabSize) = 0;
        virtual void freeAllMem() = 0;
        virtual void allocateBuffer(BufferList::iterator segIt, const size_t pageSize, const size_t numBytes) = 0;
        std::recursive_mutex globalMutex_;  // hack for now - lets profile this to see impact on performance - may not matter given the workload
        std::mutex bufferIdMutex_;  
        
        //std::map<ChunkKey, Buffer*> chunkIndex_;
        std::map<ChunkKey, BufferList::iterator> chunkIndex_;
        size_t maxBufferSize_;   /// max number of bytes allocated for the buffer poo
        size_t slabSize_;   /// size of the individual memory allocations that compose the buffer pool (up to maxBufferSize_)
        size_t maxNumSlabs_;
        size_t pageSize_;
        unsigned int bufferEpoch_;
        AbstractBufferMgr *parentMgr_;
        int maxBufferId_;
        //File_Namespace::FileMgr *fileMgr_;

        /// Maps sizes of free memory areas to host buffer pool memory addresses
        //@todo change this to multimap
        //std::multimap<size_t, int8_t *> freeMem_;
        BufferList unsizedSegs_;
        //std::map<size_t, int8_t *> freeMem_;

        BufferList::iterator evict(BufferList::iterator &evictStart, const size_t numPagesRequested, const int slabNum);
        BufferList::iterator findFreeBuffer(size_t numBytes);

        /**
         * @brief Gets a buffer of required size and returns an iterator to it
         *
         * If possible, this function will just select a free buffer of
         * sufficient size and use that. If not, it will evict as many
         * non-pinned but used buffers as needed to have enough space for the
         * buffer
         *
         * @return An iterator to the reserved buffer. We guarantee that this
         * buffer won't be evicted by PINNING it - caller should change this to
         * USED if applicable
         *
         */


    };

} // Buffer_Namespace

#endif // DATAMGR_MEMORY_BUFFER_BUFFERMGR_H
