/*
 * @file    MapDServer.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef GPUSORTIMPL_H
#define GPUSORTIMPL_H

#include <stdint.h>

void sort_on_device(int64_t* val_buff, int64_t* key_buff, const uint64_t entry_count, const bool desc);
void apply_permutation_on_device(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, int64_t* tmp_buff);

#endif  // GPUSORTIMPL_H
