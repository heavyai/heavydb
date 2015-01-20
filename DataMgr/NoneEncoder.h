#ifndef NONE_ENCODER_H
#define NONE_ENCODER_H

#include "AbstractBuffer.h"
#include "Encoder.h"

template <typename T>
class NoneEncoder : public Encoder {

    public:
        NoneEncoder(Data_Namespace::AbstractBuffer *buffer): Encoder(buffer), min_(std::numeric_limits<T>::max()),max_(std::numeric_limits<T>::min()) {}

        void appendData(mapd_addr_t srcData, const mapd_size_t numAppendElems) {
            T * unencodedData = reinterpret_cast<T *> (srcData); 
            for (mapd_size_t i = 0; i < numAppendElems; ++i) {
                min_ = std::min(min_,unencodedData[i]);
                max_ = std::max(max_,unencodedData[i]);
            }
            std::cout << "min_ " << min_ << std::endl;
            std::cout << "max_ " << max_ << std::endl;
            numElems += numAppendElems;
            buffer_ -> append(srcData,numAppendElems*sizeof(T));
        }

        void writeMetadata(FILE *f) {
            // assumes pointer is already in right place
            fwrite((mapd_addr_t)&numElems,sizeof(mapd_size_t),1,f); 
            fwrite((mapd_addr_t)&min_,sizeof(T),1,f); 
            fwrite((mapd_addr_t)&max_,sizeof(T),1,f); 
        }

        void readMetadata(FILE *f) {
            // assumes pointer is already in right place
            std::cout << "Reading metadata for none encoding" << std::endl;
            fread((mapd_addr_t)&numElems,sizeof(mapd_size_t),1,f); 
            fread((mapd_addr_t)&min_,1,sizeof(T),f); 
            fread((mapd_addr_t)&max_,1,sizeof(T),f); 
        }

    private:
        T min_;
        T max_;


}; // class NoneEncoder

#endif // NONE_ENCODER_H
