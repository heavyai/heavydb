/**
 * @file    AbstractBufferMgr.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_ABSTRACTDATAMGR_H
#define DATAMGR_MEMORY_ABSTRACTDATAMGR_H

#include "../Shared/types.h"
#include "AbstractBuffer.h"

enum MgrType {FILE_MGR, CPU_MGR, GPU_MGR};   

namespace Memory_Namespace {

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
        virtual AbstractBuffer* createChunk(const ChunkKey &key, const mapd_size_t pageSize = 0, const mapd_size_t initialSize = 0) = 0;
        virtual void deleteChunk(const ChunkKey &key) = 0;
        virtual AbstractBuffer* getChunk(ChunkKey &key, const mapd_size_t numBytes = 0) = 0;
        virtual void fetchChunk(const ChunkKey &key, AbstractBuffer *destBuffer, const mapd_size_t numBytes = 0) = 0;
        //virtual AbstractBuffer* putChunk(const ChunkKey &key, AbstractBuffer *srcBuffer, const mapd_size_t numBytes = 0) = 0;
        virtual AbstractBuffer* putChunk(const ChunkKey &key, AbstractBuffer *srcBuffer, const mapd_size_t numBytes = 0) = 0;
        virtual void checkpoint() = 0;

        // Buffer API
        virtual AbstractBuffer* createBuffer(mapd_size_t pageSize, mapd_size_t numBytes) = 0;
        virtual void deleteBuffer(AbstractBuffer *d) = 0;
        virtual AbstractBuffer* putBuffer(AbstractBuffer *d) = 0;
        virtual MgrType getMgrType() = 0;

    protected:
        AbstractBufferMgr * parentMgr_;


    };
    
} // Memory_Namespace

#endif // DATAMGR_MEMORY_ABSTRACTDATAMGR_H
