/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "PPLL/ppllCommon.h"
//</includes>
// FRAGMENT SHADER
#define numSamples <numSamples>
#define numTiles <numTiles>
#define tileIndexOffset <tileIndexOffset>

layout(location = 0) flat in uint32_t fPolygonID;

// Per pixel fragment counts
layout(r32ui) uniform restrict readonly uimage2D fragment_count_image;

// Uniforms
layout(std430, binding = 4) uniform IMAGE_TILES_UBO {
  Tile tiles[numTiles];
};

// Fragment slot allocation atomic
layout(std430) buffer PPLL_ALLOC_SSBO {
  uint32_t record_count;
};

// Fragment buffer
coherent layout (std430) buffer PPLL_RECORDS_SSBO {
  uint64_t records[];
};

layout(push_constant) uniform PPLL_CAPTURE_FRAG_PUSH_CONSTANTS {
  layout(offset = tileIndexOffset) uint32_t tileIndex;
} pushConstants;

// "Allocate" an index to use for the next record. Reserve width*height
// records for the list head record
uint32_t allocIndex(uint32_t list_head_region_size) {
  return list_head_region_size + atomicAdd(record_count, 1);
}

// sample mask, polygon id, next index
#if numSamples == 2
#define PACK_RECORD(sm, id, ni) (((uint64_t(sm) & 0x03) << 62) | ((uint64_t(id) & 0x3FFFFFFF) << 32) | (uint64_t(ni)))
#define SM_BITS(x)              (int(x) >> 30)
#define COUNT_MASK(x)           ((x) & 0x3FFFFFFFU)
#elif numSamples == 4
#define PACK_RECORD(sm, id, ni) (((uint64_t(sm) & 0x0F) << 60) | ((uint64_t(id) & 0x0FFFFFFF) << 32) | (uint64_t(ni)))
#define SM_BITS(x)              (int(x) >> 28)
#define COUNT_MASK(x)           ((x) & 0x0FFFFFFFU)
#elif numSamples == 8
#define PACK_RECORD(sm, id, ni) (((uint64_t(sm) & 0xFF) << 56) | ((uint64_t(id) & 0x00FFFFFF) << 32) | (uint64_t(ni)))
#define SM_BITS(x)              (int(x) >> 24)
#define COUNT_MASK(x)           ((x) & 0x00FFFFFFU)
#endif

void main(void) {
  ivec2 P = ivec2(gl_FragCoord.xy);
  uint32_t fragment_count = imageLoad(fragment_count_image, P).r;
  uint32_t tile_index = pushConstants.tileIndex;
  uint32_t head_index = uint32_t(P.x - tiles[tile_index].x
                             + ((P.y - tiles[tile_index].y) * tiles[tile_index].w));
  uint32_t list_head_region_size = tiles[tile_index].w * tiles[tile_index].h;

  // strip already completed samples from the sample mask (batching)
  int out_sample_mask = gl_SampleMaskIn[0] & ~SM_BITS(fragment_count);
  // Check if any samples for this fragment are incomplete
  if (out_sample_mask != 0) {
    // Only grab a slot for "next" if the pixel has more than one fragment
    if (COUNT_MASK(fragment_count) > 1) {
      uint32_t new_index = allocIndex(list_head_region_size);

      uint64_t record = PACK_RECORD(out_sample_mask, fPolygonID, new_index);

      uint64_t old_head_record = atomicExchange(records[head_index], record);
      records[new_index] = old_head_record;
    } else {
      // Only 1 fragment, just use the head record
      records[head_index] = PACK_RECORD(out_sample_mask, fPolygonID, 0);
    }
  }
}
