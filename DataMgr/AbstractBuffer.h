/**
 * @file    AbstractBuffer.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_ABSTRACTBUFFER_H
#define DATAMGR_MEMORY_ABSTRACTBUFFER_H

#include "../Shared/types.h"
#include "../Shared/sqltypes.h"
#include "MemoryLevel.h"
#include "Encoder.h"


#ifdef BUFFER_MUTEX
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#endif

namespace Data_Namespace {
    
    /**
     * @class   AbstractBuffer
     * @brief   An AbstractBuffer is a unit of data management for a data manager.
     */

    //enum BufferType {FILE_BUFFER, CPU_BUFFER, GPU_BUFFER};

    class AbstractBuffer {
        
    public:

        AbstractBuffer (const int deviceId): encoder(0), hasEncoder(0), size_(0),  isDirty_(false),isAppended_(false),isUpdated_(false), deviceId_(deviceId) {}
        AbstractBuffer (const int deviceId, const SQLTypeInfo sqlType, const EncodingType encodingType=kENCODING_NONE, const int numEncodingBits=0): size_(0),isDirty_(false),isAppended_(false),isUpdated_(false), deviceId_(deviceId){
        initEncoder(sqlType, encodingType, numEncodingBits);
        }
        virtual ~AbstractBuffer() { if (hasEncoder) delete encoder; }
        
        virtual void read(int8_t * const dst, const size_t numBytes, const size_t offset = 0, const MemoryLevel dstBufferType = CPU_LEVEL, const int dstDeviceId = -1) = 0;
        virtual void write(int8_t * src, const size_t numBytes, const size_t offset = 0, const MemoryLevel srcBufferType = CPU_LEVEL, const int srcDeviceId = -1) = 0;
        virtual void reserve(size_t numBytes) = 0;
        virtual void append(int8_t * src, const size_t numBytes, const MemoryLevel srcBufferType = CPU_LEVEL, const int deviceId = -1) = 0;
        virtual int8_t* getMemoryPtr() = 0;
        
        virtual size_t pageCount() const = 0;
        virtual size_t pageSize() const = 0;
        virtual size_t size() const = 0;
        virtual size_t reservedSize() const = 0;
        //virtual size_t used() const = 0;
        virtual int getDeviceId() const {return deviceId_;}
        virtual MemoryLevel getType() const = 0;

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

        void setSize(const size_t size) {
            size_ = size;
        }
        void clearDirtyBits() {
            isAppended_ = false;
            isUpdated_ = false;
            isDirty_ = false;
        }
        void initEncoder(const SQLTypeInfo tmpSqlType, const EncodingType tmpEncodingType = kENCODING_NONE, const int tmpEncodingBits = 0) {
            hasEncoder = true;
            sqlType = tmpSqlType;
            encodingType = tmpEncodingType;
            encodingBits = tmpEncodingBits;
            encoder = Encoder::Create(this,sqlType,encodingType,encodingBits);
        }

        void syncEncoder(const AbstractBuffer *srcBuffer) {
            hasEncoder = srcBuffer->hasEncoder;
            if (hasEncoder) {
                if (encoder == 0) { // Encoder not initialized
                    initEncoder(srcBuffer->sqlType,srcBuffer->encodingType,srcBuffer->encodingBits);
                }
                encoder->copyMetadata(srcBuffer->encoder);
            }
        }



        Encoder * encoder;
        bool hasEncoder;
        SQLTypeInfo sqlType;
        EncodingType encodingType;
        int encodingBits;
        //EncodedDataType encodedDataType;

    protected:

        size_t size_;
        bool isDirty_;
        bool isAppended_;
        bool isUpdated_;
        int deviceId_;

#ifdef BUFFER_MUTEX
        boost::shared_mutex readWriteMutex_;
        boost::shared_mutex appendMutex_;
#endif

    };
    
} // Data_Namespace

#endif // DATAMGR_MEMORY_ABSTRACTBUFFER_H
