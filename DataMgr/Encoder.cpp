#include "Encoder.h"
#include "NoneEncoder.h"
#include "FixedLengthEncoder.h"


Encoder * Encoder::Create(Data_Namespace::AbstractBuffer *buffer, const SQLTypes sqlType, const EncodingType encodingType, const EncodedDataType encodedDataType) {
    std::cout << "Encoding type: " << encodingType << std::endl;
    std::cout << "Sql type: " << sqlType << std::endl;
    switch (encodingType) {
        case kENCODING_NONE: {
            switch (sqlType) {
                case kCHAR: {
                    return new NoneEncoder <char>  (buffer);
                    break;
                }
                case kSMALLINT: {
                    return new NoneEncoder <int16_t>  (buffer);
                    break;
                }
                case kINT: {
                    return new NoneEncoder <int32_t>  (buffer);
                    break;
                }
                case kBIGINT: {
                    return new NoneEncoder <int64_t>  (buffer);
                    break;
                }
                case kFLOAT: {
                    std::cout << "Making NoneEncoder float" << std::endl;
                    return new NoneEncoder <float>  (buffer);
                    break;
                }
                case kDOUBLE: {
                    return new NoneEncoder <double>  (buffer);
                    break;
                }
            }
            break;
         }
        case kENCODING_FIXED: {
            switch (sqlType) {
                case kSMALLINT: {
                    switch(encodedDataType) {
                        case kINT8:
                            return new FixedLengthEncoder <int16_t,int8_t>  (buffer);
                            break;
                        case kINT16:
                            return new FixedLengthEncoder <int16_t,int16_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                break;
                }
                case kINT: {
                    switch(encodedDataType) {
                        case kINT8:
                            return new FixedLengthEncoder <int32_t,int8_t> (buffer);
                            break;
                        case kINT16:
                            return new FixedLengthEncoder <int32_t,int16_t> (buffer);
                            break;
                        case kINT32:
                            return new FixedLengthEncoder <int32_t,int32_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                }
                break;
                case kBIGINT: {
                    switch(encodedDataType) {
                        case kINT8:
                            return new FixedLengthEncoder <int64_t,int8_t> (buffer);
                            break;
                        case kINT16:
                            return new FixedLengthEncoder <int64_t,int16_t> (buffer);
                            break;
                        case kINT32:
                            return new FixedLengthEncoder <int64_t,int32_t> (buffer);
                            break;
                        case kINT64:
                            return new FixedLengthEncoder <int64_t,int64_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                break;
                }
            } // switch (sqlType)
            break;
        } // Case: kENCODING_FIXED
        default:
            return 0;
            break;
    } // switch (encodingType)

}

