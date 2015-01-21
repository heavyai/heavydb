#ifndef FIXED_LENGTH_ENCODER_H
#define FIXED_LENGTH_ENCODER_H
#include "Encoder.h"
#include "AbstractBuffer.h"
#include <stdexcept>
#include <iostream>

template <typename T, typename V>
class FixedLengthEncoder : public Encoder {

    public:
        FixedLengthEncoder(Data_Namespace::AbstractBuffer *buffer): Encoder(buffer), dataMin(std::numeric_limits<T>::max()),dataMax(std::numeric_limits<T>::min()) {}

        ChunkMetadata appendData(mapd_addr_t srcData, const mapd_size_t numAppendElems) {
            T * unencodedData = reinterpret_cast<T *> (srcData); 
            V * encodedData = new V [numAppendElems];  
            for (mapd_size_t i = 0; i < numAppendElems; ++i) {
                //std::cout << "Unencoded: " << unencodedData[i] << std::endl;
                //std::cout << "Min: " << dataMin << " Max: " <<  dataMax << std::endl;
                encodedData[i] = static_cast <V>(unencodedData[i]);
                if (unencodedData[i] != encodedData[i]) {
                    std::cout << "Unencoded: " << unencodedData[i] << " Encoded: " << encodedData[i] << std::endl;
                    delete [] encodedData;
                    throw std::runtime_error ("Encoding failed");
                }
                else {
                    dataMin = std::min(dataMin,unencodedData[i]);
                    dataMax = std::max(dataMax,unencodedData[i]);
                }

            }
            numElems += numAppendElems;
            std::cout << "Min: " << dataMin << " Max: " << dataMax << std::endl;

            // assume always CPU_BUFFER?
            buffer_ -> append((mapd_addr_t)(encodedData),numAppendElems*sizeof(V));
            delete [] encodedData;
            ChunkMetadata chunkMetadata;
            getMetadata(chunkMetadata);
            return chunkMetadata;
        }





        void getMetadata(ChunkMetadata &chunkMetadata) {
            Encoder::getMetadata(chunkMetadata); // call on parent class
            chunkMetadata.fillChunkStats(dataMin,dataMax);
        }

        void copyMetadata(const Encoder * copyFromEncoder) {
            numElems = copyFromEncoder -> numElems;
            auto castedEncoder = reinterpret_cast <const FixedLengthEncoder <T, V> *> (copyFromEncoder);
            dataMin = castedEncoder -> dataMin;
            dataMax = castedEncoder -> dataMax;
        }


        void writeMetadata(FILE *f) {
            // assumes pointer is already in right place
            fwrite((mapd_addr_t)&numElems,sizeof(mapd_size_t),1,f); 
            fwrite((mapd_addr_t)&dataMin,sizeof(T),1,f); 
            fwrite((mapd_addr_t)&dataMax,sizeof(T),1,f); 
        }

        void readMetadata(FILE *f) {
            // assumes pointer is already in right place
            fread((mapd_addr_t)&numElems,sizeof(mapd_size_t),1,f); 
            fread((mapd_addr_t)&dataMin,1,sizeof(T),f); 
            fread((mapd_addr_t)&dataMax,1,sizeof(T),f); 
        }
        T dataMin;
        T dataMax;

}; // FixedLengthEncoder

#endif // FIXED_LENGTH_ENCODER_H
