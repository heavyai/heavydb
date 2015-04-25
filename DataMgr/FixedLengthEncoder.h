#ifndef FIXED_LENGTH_ENCODER_H
#define FIXED_LENGTH_ENCODER_H
#include "Encoder.h"
#include "AbstractBuffer.h"
#include <stdexcept>
#include <iostream>
#include <memory>
#include <glog/logging.h>

template <typename T, typename V>
class FixedLengthEncoder : public Encoder {

    public:
        FixedLengthEncoder(Data_Namespace::AbstractBuffer *buffer): Encoder(buffer), dataMin(std::numeric_limits<T>::max()),dataMax(std::numeric_limits<T>::min()) {}

        ChunkMetadata appendData(int8_t * &srcData, const size_t numAppendElems) {
            T * unencodedData = reinterpret_cast<T *> (srcData); 
            std::unique_ptr<V> encodedData = std::unique_ptr<V>(new V [numAppendElems]);  
            for (size_t i = 0; i < numAppendElems; ++i) {
                //std::cout << "Unencoded: " << unencodedData[i] << std::endl;
                //std::cout << "Min: " << dataMin << " Max: " <<  dataMax << std::endl;
                encodedData.get()[i] = static_cast <V>(unencodedData[i]);
                if (unencodedData[i] != encodedData.get()[i]) {
                    LOG(ERROR) << "Fixed encoding failed, Unencoded: " + std::to_string(unencodedData[i]) + " encoded: " + std::to_string(encodedData.get()[i]);
                }
                else {
                    dataMin = std::min(dataMin,unencodedData[i]);
                    dataMax = std::max(dataMax,unencodedData[i]);
                }

            }
            numElems += numAppendElems;

            // assume always CPU_BUFFER?
            buffer_ -> append((int8_t *)(encodedData.get()),numAppendElems*sizeof(V));
            ChunkMetadata chunkMetadata;
            getMetadata(chunkMetadata);
            srcData += numAppendElems *sizeof(T);
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
            CHECK_RET(fwrite((int8_t *)&numElems,sizeof(size_t),1,f));
            CHECK_RET(fwrite((int8_t *)&dataMin,sizeof(T),1,f));
            CHECK_RET(fwrite((int8_t *)&dataMax,sizeof(T),1,f));
        }

        void readMetadata(FILE *f) {
            // assumes pointer is already in right place
            CHECK_RET(fread((int8_t *)&numElems,sizeof(size_t),1,f));
            CHECK_RET(fread((int8_t *)&dataMin,1,sizeof(T),f));
            CHECK_RET(fread((int8_t *)&dataMax,1,sizeof(T),f));
        }
        T dataMin;
        T dataMax;

}; // FixedLengthEncoder

#endif // FIXED_LENGTH_ENCODER_H
