#ifndef BITMAP_GENERATORS_h
#define BITMAP_GENERATORS_h

#include <cstddef>
#include <cstdint>

// IMPORTANT NOTE: All function,  generating bitmaps, assume that the
//  allocated `*src' memory has size that is multiple of 64 bytes.

size_t gen_null_bitmap_8(uint8_t* bitmap,
                         const uint8_t* data,
                         size_t size,
                         const uint8_t null_val);
size_t gen_null_bitmap_16(uint8_t* bitmap,
                          const uint16_t* data,
                          size_t size,
                          const uint16_t null_val);
size_t gen_null_bitmap_32(uint8_t* bitmap,
                          const uint32_t* data,
                          size_t size,
                          const uint32_t null_val);
size_t gen_null_bitmap_64(uint8_t* bitmap,
                          const uint64_t* data,
                          size_t size,
                          const uint64_t null_val);
#endif