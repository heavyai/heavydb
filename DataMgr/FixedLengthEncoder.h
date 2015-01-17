#ifndef FIXEDLENGTHENCODER_H
#define FIXEDLENGTHENCODER_H
#include "Encoder.h"
#include "AbstractBuffer.h"
#include <stdexcept>
#include <iostream>

template <typename T, typename V>
class FixedLengthEncoder : public Encoder {

    public:
        FixedLengthEncoder(Memory_Namespace::AbstractBuffer *buffer): Encoder(buffer), min_(std::numeric_limits<T>::max()),max_(std::numeric_limits<T>::min()) {}

        void appendData(mapd_addr_t srcData, const mapd_size_t numAppendElems) {
            T * unencodedData = reinterpret_cast<T *> (srcData); 
            V * encodedData = new V [numAppendElems];  
            for (mapd_size_t i = 0; i < numAppendElems; ++i) {
                encodedData[i] = static_cast <V>(unencodedData[i]);
                min_ = std::min(min_,unencodedData[i]);
                max_ = std::max(min_,unencodedData[i]);
                if (unencodedData[i] != encodedData[i]) {
                    delete [] encodedData;
                    throw std::runtime_error ("Encoding failed");
                }
            }
            numElems += numAppendElems;
            std::cout << "Min: " << min_ << " Max: " << max_ << std::endl;

            // assume always CPU_BUFFER?
            buffer_ -> append((mapd_addr_t)(encodedData),numAppendElems*sizeof(V));
            delete [] encodedData;
        }

        void writeMetadata(FILE *f/*, const mapd_size_t offset*/) {
            // assumes pointer is already in right place

            //fseek(f, offset, SEEK_SET);
            fwrite((mapd_addr_t)&numElems,sizeof(mapd_size_t),1,f); 
            fwrite((mapd_addr_t)&min_,sizeof(T),1,f); 
            fwrite((mapd_addr_t)&max_,sizeof(T),1,f); 
        }

        void readMetadata(FILE *f /*, const mapd_size_t offset*/) {

            //fseek(f, offset, SEEK_SET);
            fread((mapd_addr_t)&numElems,sizeof(mapd_size_t),1,f); 
            fread((mapd_addr_t)&min_,1,sizeof(T),f); 
            fread((mapd_addr_t)&max_,1,sizeof(T),f); 
        }


    private:
        T min_;
        T max_;
        //AbstractBuffer * buffer_;
        //mapd_size_t rawElementWidth_;
        //mapd_size_t encodedElementWidth_;
}; // FixedLengthEncoder

#endif // FIXEDLENGTHENCODER_H
