#ifndef ENCODETYPES_H
#define ENCODETYPES_H

enum EncodingAlgo {
    NONE = 0,
    FIXED = 1,
    DIFFERENTIAL = 2
};

enum EncodingType {
    kINT8 = 0,
    kINT16 = 1,
    kINT32 = 2,
    kINT64 = 3,
    kUINT8 = 4,
    kUINT16 = 5,
    kUINT32 = 6,
    kUINT64 = 7
};

#endif // ENCODETYPES_H
