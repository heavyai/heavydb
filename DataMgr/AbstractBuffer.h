/**
 * @file    AbstractBuffer.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_ABSTRACTBUFFER_H
#define DATAMGR_MEMORY_ABSTRACTBUFFER_H

#include "../Shared/types.h"

#ifdef BUFFER_MUTEX
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#endif

namespace Memory_Namespace {
    
    /**
     * @class   AbstractBuffer
     * @brief   An AbstractBuffer is a unit of data management for a data manager.
     */

    enum BufferType {FILE_BUFFER, CPU_BUFFER, GPU_BUFFER};

    class AbstractBuffer {
        
    public:

        AbstractBuffer (): size_(0),isDirty_(false),isAppended_(false),isUpdated_(false) {}
        virtual ~AbstractBuffer() {}
        
        virtual void read(mapd_addr_t const dst, const mapd_size_t numBytes, const BufferType dstBufferType = CPU_BUFFER, const mapd_size_t offset = 0) = 0;
        virtual void write(mapd_addr_t src, const mapd_size_t numBytes, const BufferType srcBufferType = CPU_BUFFER, const mapd_size_t offset = 0) = 0;
        virtual void reserve(mapd_size_t numBytes) = 0;
        virtual void append(mapd_addr_t src, const mapd_size_t numBytes, const BufferType srcBufferType = CPU_BUFFER) = 0;
        virtual mapd_byte_t* getMemoryPtr() = 0;
        
        virtual mapd_size_t pageCount() const = 0;
        virtual mapd_size_t pageSize() const = 0;
        virtual mapd_size_t size() const = 0;
        virtual mapd_size_t reservedSize() const = 0;
        //virtual mapd_size_t used() const = 0;
        virtual int getDeviceId() const {return -1;}
        virtual BufferType getType() const = 0;

        // Next three methods are dummy methods so FileBuffer does not implement these
        virtual inline int pin() {return 0;}
        virtual inline int unPin() {return 0;}
        virtual inline int getPinCount() {return 0;}

        virtual inline bool isDirty() const {return isDirty_;}
        virtual inline bool isAppended() const {return isAppended_;}
        virtual inline bool isUpdated() const {return isUpdated_;}
        virtual inline void setUpdated() {
            isUpdated_ = true;
            isDirty_ = true;
        }

        virtual inline void setAppended() {
            isAppended_ = true;
            isDirty_ = true;
        }

        void setSize(const mapd_size_t size) {
            size_ = size;
        }
        void clearDirtyBits() {
            isAppended_ = false;
            isUpdated_ = false;
            isDirty_ = false;
        }


    protected:
        mapd_size_t size_;
        bool isDirty_;
        bool isAppended_;
        bool isUpdated_;

#ifdef BUFFER_MUTEX
        boost::shared_mutex readWriteMutex_;
        boost::shared_mutex appendMutex_;
#endif

    };
    
} // Memory_Namespace

#endif // DATAMGR_MEMORY_ABSTRACTBUFFER_H
