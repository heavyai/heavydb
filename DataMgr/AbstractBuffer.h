/**
 * @file    AbstractBuffer.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_ABSTRACTBUFFER_H
#define DATAMGR_MEMORY_ABSTRACTBUFFER_H

#include "../Shared/types.h"

namespace Memory_Namespace {
    
    /**
     * @class   AbstractBuffer
     * @brief   An AbstractBuffer is a unit of data management for a data manager.
     */
    class AbstractBuffer {
        
    public:

        AbstractBuffer (): size_(0),isDirty_(false),isAppended_(false),isUpdated_(false) {}
        virtual ~AbstractBuffer() {}
        
        virtual void read(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset = 0) = 0;
        virtual void write(mapd_addr_t src, const mapd_size_t numBytes, const mapd_size_t offset = 0) = 0;
        virtual void reserve(mapd_size_t numBytes) = 0;
        virtual void append(mapd_addr_t src, const mapd_size_t numBytes) = 0;
        virtual mapd_byte_t* getMemoryPtr() = 0;
        
        virtual mapd_size_t pageCount() const = 0;
        virtual mapd_size_t pageSize() const = 0;
        virtual mapd_size_t size() const = 0;
        virtual mapd_size_t reservedSize() const = 0;
        //virtual mapd_size_t used() const = 0;
        virtual bool isDirty() const = 0;
        void setSize(const mapd_size_t size) {
            size_ = size;
        }

    protected:
        mapd_size_t size_;
        bool isDirty_;
        bool isAppended_;
        bool isUpdated_;

    };
    
} // Memory_Namespace

#endif // DATAMGR_MEMORY_ABSTRACTBUFFER_H
