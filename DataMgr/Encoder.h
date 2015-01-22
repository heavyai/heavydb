#ifndef ENCODER_H
#define ENCODER_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <limits>
#include "../Shared/types.h"
#include "../Shared/sqltypes.h"

namespace Data_Namespace {
    class AbstractBuffer;
}

struct ChunkStats {
    Datum min;
    Datum max;
};

struct ChunkMetadata {
    SQLTypes sqlType;
    EncodingType encodingType;
    int encodingBits;
    int64_t numBytes;
    int64_t numElements;
    ChunkStats chunkStats;

    template <typename T> void fillChunkStats (const T min, const T max) {
        switch (sqlType) {
            case kSMALLINT: {
                chunkStats.min.smallintval = min;
                chunkStats.max.smallintval = max;
                break;
            }
            case kINT: {
                chunkStats.min.intval = min;
                chunkStats.max.intval = max;
                break;
            }
            case kBIGINT: {
                chunkStats.min.bigintval = min;
                chunkStats.max.bigintval = max;
                break;
            }
            case kFLOAT: {
                chunkStats.min.floatval = min;
                chunkStats.max.floatval = max;
                break;
            }
            case kDOUBLE: {
                chunkStats.min.doubleval = min;
                chunkStats.max.doubleval = max;
                break;
            }
            default: {
                break;
            }
        }
    }
};

class Encoder {
    public: 
        static Encoder * Create(Data_Namespace::AbstractBuffer * buffer, const SQLTypes sqlType, const EncodingType encodingType, const int encodingBits);
        Encoder(Data_Namespace::AbstractBuffer * buffer): buffer_(buffer), numElems(0) {
            /*
            metadataTemplate_.sqlType = buffer_ -> sqlType;
            metadataTemplate_.encodingType = buffer_ -> encodingType;
            metadataTemplate_.encodingBits = buffer_ -> encodingBits;
            */
        }
        virtual ChunkMetadata appendData(mapd_addr_t srcData, const mapd_size_t numAppendElems) = 0;
        virtual void getMetadata (ChunkMetadata &chunkMetadata); 
        virtual void copyMetadata (const Encoder * copyFromEncoder) = 0; 
        virtual void writeMetadata(FILE *f/*, const mapd_size_t offset*/) = 0;
        virtual void readMetadata(FILE *f/*, const mapd_size_t offset*/) = 0;
        mapd_size_t numElems;

    protected:
        Data_Namespace::AbstractBuffer * buffer_;
        //ChunkMetadata metadataTemplate_;
};


#endif // Encoder_h
