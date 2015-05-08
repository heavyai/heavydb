#ifndef CHUNKMETADATA_H
#define CHUNKMETADATA_H

#include "../Shared/sqltypes.h"
#include <stddef.h>

struct ChunkStats {
    Datum min;
    Datum max;
    bool has_nulls;
};

struct ChunkMetadata {
    SQLTypeInfo sqlType;
    size_t numBytes;
    size_t numElements;
    ChunkStats chunkStats;

    template <typename T> void fillChunkStats (const T min, const T max, const bool has_nulls) {
        chunkStats.has_nulls = has_nulls;
        switch (sqlType.get_type()) {
            case kBOOLEAN: {
                chunkStats.min.tinyintval = min;
                chunkStats.max.tinyintval = max;
                break;
            }
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
            case kVARCHAR:
            case kCHAR:
            case kTEXT:
              if (sqlType.get_compression() == kENCODING_DICT) {
                chunkStats.min.intval = min;
                chunkStats.max.intval = max;
              }
              break;
            default: {
                break;
            }
        }
    }
};

#endif //CHUNKMETADATA_H
