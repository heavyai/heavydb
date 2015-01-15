#include "Encoder.h"
#include "FixedLengthEncoder.h"


Encoder * Encoder::Create(Memory_Namespace::AbstractBuffer *buffer, const SQLTypes sqlType, const EncodingType encodingType, const EncodedDataType encodedDataType) {
    switch (encodingType) {
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

        } // Case: kENCODING_FIXED
        break;
        default:
            return 0;
            break;
    } // switch (compressionType)
}

