/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sqltypes.h"

namespace shared {

// Same as strftime(buf, max, "%F", tm) but guarantees that the year is
// zero-padded to a minimum length of 4. Return the number of characters
// written, not including null byte. If max is not large enough, return 0.
void computeDate(int64_t const unixtime,
                 uint32_t& y_out,
                 uint32_t& m_out,
                 uint32_t& d_out);
size_t formatDate(char* buf, size_t const max, int64_t const unixtime);

// Same as strftime(buf, max, "%F %T", tm) but guarantees that the year is
// zero-padded to a minimum length of 4. Return the number of characters
// written, not including null byte. If max is not large enough, return 0.
// Requirement: 0 <= dimension <= 9.
void computeDateTime(const int64_t timestamp,
                     const int32_t dimension,
                     uint32_t& y_out,
                     uint32_t& m_out,
                     uint32_t& d_out,
                     uint32_t& hh_out,
                     uint32_t& mm_out,
                     uint32_t& ss_out,
                     uint32_t& frac_out);
size_t formatDateTime(char* buf,
                      size_t const max,
                      int64_t const timestamp,
                      int const dimension,
                      bool use_iso_format = false);

// Write unixtime in seconds since epoch as "HH:MM:SS" format.
void computeHMS(const int64_t unixtime,
                uint32_t& hh_out,
                uint32_t& mm_out,
                uint32_t& ss_out);
size_t formatHMS(char* buf, size_t const max, int64_t const unixtime);

// Write unix time in seconds since epoch as ISO 8601 format for the given temporal type.
std::string convert_temporal_to_iso_format(const int64_t unix_time,
                                           const SQLTypes sql_type,
                                           const int precision);

}  // namespace shared
