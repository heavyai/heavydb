#ifndef ENCODER_H
#define ENCODER_H

#include "ChunkMetadata.h"
#include "../Shared/types.h"
#include "../Shared/sqltypes.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <limits>

namespace Data_Namespace {
    class AbstractBuffer;
}

class Encoder {
    public: 
        static Encoder * Create(Data_Namespace::AbstractBuffer * buffer, const SQLTypes sqlType, const EncodingType encodingType, const int encodingBits);
        Encoder(Data_Namespace::AbstractBuffer * buffer): numElems(0), buffer_(buffer) {
            /*
            metadataTemplate_.sqlType = buffer_ -> sqlType;
            metadataTemplate_.encodingType = buffer_ -> encodingType;
            metadataTemplate_.encodingBits = buffer_ -> encodingBits;
            */
        }
        virtual ChunkMetadata appendData(int8_t * &srcData, const size_t numAppendElems) = 0;
        virtual void getMetadata (ChunkMetadata &chunkMetadata); 
        virtual void copyMetadata (const Encoder * copyFromEncoder) = 0; 
        virtual void writeMetadata(FILE *f/*, const size_t offset*/) = 0;
        virtual void readMetadata(FILE *f/*, const size_t offset*/) = 0;
        size_t numElems;

    protected:
        Data_Namespace::AbstractBuffer * buffer_;
        //ChunkMetadata metadataTemplate_;
};


#endif // Encoder_h
