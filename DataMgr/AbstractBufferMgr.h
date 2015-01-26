/**
 * @file    AbstractBufferMgr.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#ifndef ABSTRACTDATAMGR_H
#define ABSTRACTDATAMGR_H

#include "../Shared/types.h"
#include "AbstractBuffer.h"

enum MgrType {FILE_MGR, CPU_MGR, GPU_MGR};   

namespace Data_Namespace {

    /**
     * @class   AbstractBufferMgr
     * @brief   Abstract prototype (interface) for a data manager.
     *
     * A data manager provides a common interface by inheriting the public interface
     * of this class, whose methods are pure virtual, enforcing each class that
     * implements this interfact to implement the necessary methods.
     *
     * A data manager literally manages data. One assumption about the data manager
     * is that it divides up its data into buffers of data of some kind, each of which
     * inherit the interface specified in AbstractBuffer (@see AbstractBuffer).
     */
    class AbstractBufferMgr {

    public:
        virtual ~AbstractBufferMgr() {}
        
        // Chunk API
        virtual AbstractBuffer* createChunk(const ChunkKey &key, const size_t pageSize = 0, const size_t initialSize = 0) = 0;
        virtual void deleteChunk(const ChunkKey &key) = 0;
        virtual AbstractBuffer* getChunk(const ChunkKey &key, const size_t numBytes = 0) = 0;
        virtual void fetchChunk(const ChunkKey &key, AbstractBuffer *destBuffer, const size_t numBytes = 0) = 0;
        //virtual AbstractBuffer* putChunk(const ChunkKey &key, AbstractBuffer *srcBuffer, const size_t numBytes = 0) = 0;
        virtual AbstractBuffer* putChunk(const ChunkKey &key, AbstractBuffer *srcBuffer, const size_t numBytes = 0) = 0;
        virtual void getChunkMetadataVec(std::vector<std::pair <ChunkKey,ChunkMetadata> > &chunkMetadata) = 0;
        virtual void getChunkMetadataVecForKeyPrefix(std::vector<std::pair <ChunkKey,ChunkMetadata> > &chunkMetadataVec, const ChunkKey &keyPrefix) = 0;

        virtual void checkpoint() = 0;

        // Buffer API
        virtual AbstractBuffer* createBuffer(const size_t numBytes = 0) = 0;
        virtual void deleteBuffer(AbstractBuffer *buffer) = 0;
        //virtual AbstractBuffer* putBuffer(AbstractBuffer *d) = 0;
        virtual MgrType getMgrType() = 0;
        virtual size_t getNumChunks() = 0;

    protected:
        AbstractBufferMgr * parentMgr_;


    };
    
} // Data_Namespace

#endif // ABSTRACTDATAMGR_H
