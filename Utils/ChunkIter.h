/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file ChunkIter.h
 * @brief
 *
 */

#ifndef _CHUNK_ITER_H_
#define _CHUNK_ITER_H_

#include "../Shared/funcannotations.h"
#include "../Shared/sqltypes.h"

class ChunkIter {
 public:
  SQLTypeInfo type_info;
  int8_t* second_buf;
  int8_t* current_pos;
  int8_t* start_pos;
  int8_t* end_pos;
  int skip;
  int skip_size;
  size_t num_elems;
  Datum datum;  // used to hold uncompressed value
};

void ChunkIter_reset(ChunkIter* it);
DEVICE void ChunkIter_get_next(ChunkIter* it,
                               bool uncompress,
                               VarlenDatum* vd,
                               bool* is_end);
// @brief get nth element in Chunk.  Does not change ChunkIter state
DEVICE void ChunkIter_get_nth(ChunkIter* it,
                              int nth,
                              bool uncompress,
                              VarlenDatum* vd,
                              bool* is_end);
DEVICE void ChunkIter_get_nth(ChunkIter* it, int nth, ArrayDatum* vd, bool* is_end);
DEVICE void ChunkIter_get_nth_varlen(ChunkIter* it,
                                     int nth,
                                     ArrayDatum* vd,
                                     bool* is_end);
DEVICE void ChunkIter_get_nth_varlen_notnull(ChunkIter* it,
                                             int nth,
                                             ArrayDatum* vd,
                                             bool* is_end);
DEVICE void ChunkIter_get_nth_point_coords(ChunkIter* it,
                                           int nth,
                                           ArrayDatum* vd,
                                           bool* is_end);
#endif  // _CHUNK_ITER_H_
