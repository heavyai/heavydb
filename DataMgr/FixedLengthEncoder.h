#ifndef FIXEDLENGTHENCODER_H
#define FIXEDLENGTHENCODER_H
#include "Encoder.h"
#include "AbstractBuffer.h"
#include <stdexcept>
#include <iostream>

template <typename T, typename V>
class FixedLengthEncoder : public Encoder {

    public:
        FixedLengthEncoder(Memory_Namespace::AbstractBuffer *buffer): Encoder(buffer), min_(std::numeric_limits<T>::max()),max_(std::numeric_limits<T>::min()), numElems_(0)  {}

        void appendData(mapd_addr_t srcData, const mapd_size_t numElems) {
            T * unencodedData = reinterpret_cast<T *> (srcData); 
            V * encodedData = new V [numElems];  
            for (mapd_size_t i = 0; i < numElems; ++i) {
                encodedData[i] = static_cast <V>(unencodedData[i]);
                min_ = std::min(min_,unencodedData[i]);
                max_ = std::max(min_,unencodedData[i]);
                if (unencodedData[i] != encodedData[i]) {
                    delete [] encodedData;
                    throw std::runtime_error ("Encoding failed");
                }
            }
            numElems_ += numElems;
            std::cout << "Min: " << min_ << " Max: " << max_ << std::endl;

            // assume always CPU_BUFFER?
            buffer_ -> append((mapd_addr_t)(encodedData),numElems*sizeof(V));
            delete [] encodedData;
        }
    private:
        T min_;
        T max_;
        mapd_size_t numElems_;
        //AbstractBuffer * buffer_;
        //mapd_size_t rawElementWidth_;
        //mapd_size_t encodedElementWidth_;
}; // FixedLengthEncoder

#endif // FIXEDLENGTHENCODER_H
