/*
 * @file    MapDServer.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef GPUSORT_H
#define GPUSORT_H

#include <cstdint>

void sort_groups(int64_t* val_buff, int64_t* key_buff, const uint64_t entry_count, const bool desc);
void apply_permutation(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, int64_t* tmp_buff);

#endif // GPUSORT_H
