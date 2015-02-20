#ifndef CHUNKMETADATA_H
#define CHUNKMETADATA_H

#include "../Shared/sqltypes.h"
#include <stddef.h>

struct ChunkStats {
    Datum min;
    Datum max;
};

struct ChunkMetadata {
    SQLTypeInfo sqlType;
    EncodingType encodingType;
    int encodingBits;
    size_t numBytes;
    size_t numElements;
    ChunkStats chunkStats;

    template <typename T> void fillChunkStats (const T min, const T max) {
        switch (sqlType.type) {
            case kSMALLINT: {
                chunkStats.min.smallintval = min;
                chunkStats.max.smallintval = max;
                break;
            }
            case kINT: {
                chunkStats.min.intval = min;
                chunkStats.max.intval = max;
                break;
            }
            case kBIGINT: 
						case kNUMERIC:
						case kDECIMAL: {
                chunkStats.min.bigintval = min;
                chunkStats.max.bigintval = max;
                break;
            }
            case kTIME: 
						case kTIMESTAMP:
						case kDATE: {
                chunkStats.min.timeval = min;
                chunkStats.max.timeval = max;
                break;
            }
            case kFLOAT: {
                chunkStats.min.floatval = min;
                chunkStats.max.floatval = max;
                break;
            }
            case kDOUBLE: {
                chunkStats.min.doubleval = min;
                chunkStats.max.doubleval = max;
                break;
            }
            default: {
                break;
            }
        }
    }
};

#endif //CHUNKMETADATA_H
