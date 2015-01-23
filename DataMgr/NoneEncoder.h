#ifndef NONE_ENCODER_H
#define NONE_ENCODER_H

#include "AbstractBuffer.h"
#include "Encoder.h"

template <typename T>
class NoneEncoder : public Encoder {

    public:
        NoneEncoder(Data_Namespace::AbstractBuffer *buffer): Encoder(buffer), dataMin(std::numeric_limits<T>::max()),dataMax(std::numeric_limits<T>::min()) {}

        ChunkMetadata appendData(mapd_addr_t &srcData, const mapd_size_t numAppendElems) {
            T * unencodedData = reinterpret_cast<T *> (srcData); 
            std::cout << "Unencoded data 0: " << unencodedData[0] << std::endl;
            std::cout << "Unencoded data last: " << unencodedData[numAppendElems-1] << std::endl;
            for (mapd_size_t i = 0; i < numAppendElems; ++i) {
                dataMin = std::min(dataMin,unencodedData[i]);
                dataMax = std::max(dataMax,unencodedData[i]);
            }
            std::cout << "dataMin " << dataMin << std::endl;
            std::cout << "dataMax " << dataMax << std::endl;
            numElems += numAppendElems;
            buffer_ -> append(srcData,numAppendElems*sizeof(T));
            ChunkMetadata chunkMetadata;
            getMetadata(chunkMetadata);
            srcData += numAppendElems * sizeof(T);
            return chunkMetadata;
        }

        void getMetadata(ChunkMetadata &chunkMetadata) {
            Encoder::getMetadata(chunkMetadata); // call on parent class
            chunkMetadata.fillChunkStats(dataMin,dataMax);
        }

        void writeMetadata(FILE *f) {
            // assumes pointer is already in right place
            fwrite((mapd_addr_t)&numElems,sizeof(mapd_size_t),1,f); 
            fwrite((mapd_addr_t)&dataMin,sizeof(T),1,f); 
            fwrite((mapd_addr_t)&dataMax,sizeof(T),1,f); 
        }

        void readMetadata(FILE *f) {
            // assumes pointer is already in right place
            std::cout << "Reading metadata for none encoding" << std::endl;
            fread((mapd_addr_t)&numElems,sizeof(mapd_size_t),1,f); 
            fread((mapd_addr_t)&dataMin,1,sizeof(T),f); 
            fread((mapd_addr_t)&dataMax,1,sizeof(T),f); 
        }

        void copyMetadata(const Encoder * copyFromEncoder) {
            numElems = copyFromEncoder -> numElems;
            auto castedEncoder = reinterpret_cast <const NoneEncoder <T> *> (copyFromEncoder);
            dataMin = castedEncoder -> dataMin;
            dataMax = castedEncoder -> dataMax;
        }

        T dataMin;
        T dataMax;

}; // class NoneEncoder

#endif // NONE_ENCODER_H
