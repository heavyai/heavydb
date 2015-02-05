#include "Encoder.h"
#include "NoneEncoder.h"
#include "FixedLengthEncoder.h"


Encoder * Encoder::Create(Data_Namespace::AbstractBuffer *buffer, const SQLTypeInfo sqlType, const EncodingType encodingType, const int encodingBits) {
    switch (encodingType) {
        case kENCODING_NONE: {
            switch (sqlType.type) {
                case kBOOLEAN: {
                    return new NoneEncoder <int8_t>  (buffer);
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
                case kBIGINT: 
								case kNUMERIC: 
								case kDECIMAL: {
                    return new NoneEncoder <int64_t>  (buffer);
                    break;
                }
                case kFLOAT: {
                    return new NoneEncoder <float>  (buffer);
                    break;
                }
                case kDOUBLE: {
                    return new NoneEncoder <double>  (buffer);
                    break;
                }
                default: {
                    return 0;
                }
            }
            break;
         }
        case kENCODING_FIXED: {
            switch (sqlType.type) {
                case kSMALLINT: {
                    switch(encodingBits) {
                        case 8:
                            return new FixedLengthEncoder <int16_t,int8_t>  (buffer);
                            break;
                        case 16:
                            return new NoneEncoder <int16_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                break;
                }
                case kINT: {
                    switch(encodingBits) {
                        case 8:
                            return new FixedLengthEncoder <int32_t,int8_t> (buffer);
                            break;
                        case 16:
                            return new FixedLengthEncoder <int32_t,int16_t> (buffer);
                            break;
                        case 32:
                            return new NoneEncoder <int32_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                }
                break;
                case kBIGINT: 
								case kNUMERIC:
								case kDECIMAL: {
                    switch(encodingBits) {
                        case 8:
                            return new FixedLengthEncoder <int64_t,int8_t> (buffer);
                            break;
                        case 16:
                            return new FixedLengthEncoder <int64_t,int16_t> (buffer);
                            break;
                        case 32:
                            return new FixedLengthEncoder <int64_t,int32_t> (buffer);
                            break;
                        case 64:
                            return new NoneEncoder <int64_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                break;
                }
                default: {
                    return 0;
                    break;
                }
            } // switch (sqlType)
            break;
        } // Case: kENCODING_FIXED
        default: {
            return 0;
            break;
        }
    } // switch (encodingType)
    return 0;

}

void Encoder::getMetadata(ChunkMetadata &chunkMetadata) {
    //chunkMetadata = metadataTemplate_; // invoke copy constructor
    chunkMetadata.sqlType = buffer_ -> sqlType; 
    chunkMetadata.encodingType = buffer_ -> encodingType; 
    chunkMetadata.encodingBits = buffer_ -> encodingBits; 
    chunkMetadata.numBytes = buffer_ -> size();
    chunkMetadata.numElements = numElems;
}
