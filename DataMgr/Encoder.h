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


class Encoder {
    public: 
        static Encoder * Create(Data_Namespace::AbstractBuffer * buffer, const SQLTypes sqlType, const EncodingType encodingType, const int encodingBits);
        Encoder(Data_Namespace::AbstractBuffer * buffer): buffer_(buffer), numElems(0) {}
        virtual void appendData(mapd_addr_t srcData, const mapd_size_t numAppendElems) = 0;
        virtual void copyMetadata (const Encoder * copyFromEncoder) = 0; 
        virtual void writeMetadata(FILE *f/*, const mapd_size_t offset*/) = 0;
        virtual void readMetadata(FILE *f/*, const mapd_size_t offset*/) = 0;
        mapd_size_t numElems;

    protected:
        Data_Namespace::AbstractBuffer * buffer_;
};




#endif // Encoder_h
