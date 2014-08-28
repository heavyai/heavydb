/**
 * @file    AbstractDataMgr.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_ABSTRACTDATAMGR_H
#define DATAMGR_MEMORY_ABSTRACTDATAMGR_H

#include "../../Shared/types.h"
#include "AbstractDatum.h"

namespace Memory_Namespace {

    /**
     * @class   AbstractDataMgr
     * @brief   Abstract prototype (interface) for a data manager.
     *
     * A data manager provides a common interface by inheriting the public interface
     * of this class, whose methods are pure virtual, enforcing each class that
     * implements this interfact to implement the necessary methods.
     *
     * A data manager literally manages data. One assumption about the data manager
     * is that it divides up its data into buffers of data of some kind, each of which
     * inherit the interface specified in AbstractDatum (@see AbstractDatum).
     */
    class AbstractDataMgr {

    public:
        
        virtual ~AbstractDataMgr() {}
        
        // Chunk API
        virtual AbstractDatum* createChunk(const ChunkKey &key, mapd_size_t pageSize, mapd_size_t nbytes = 0, mapd_addr_t buf = nullptr) = 0;
        virtual void deleteChunk(const ChunkKey &key) = 0;
        virtual void releaseChunk(const ChunkKey &key) = 0;
        virtual void getChunk(ChunkKey &key) = 0;
        virtual AbstractDatum* putChunk(const ChunkKey &key, AbstractDatum *d) = 0;

        // Datum API
        virtual void createDatum(mapd_size_t pageSize, mapd_size_t nbytes = 0) = 0;
        virtual void deleteDatum(int id) = 0;
        virtual AbstractDatum* putDatum(AbstractDatum *d) = 0;
    };
    
} // Memory_Namespace

#endif // DATAMGR_MEMORY_ABSTRACTDATAMGR_H
