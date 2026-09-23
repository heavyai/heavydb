/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// coord conversion functions
//

#define NULL_DECIMAL (int64_t(1) << 63)

#define SEPARATOR_INT32  -2147483647                // -std::numeric_limits<int32_t>::max()
#define SEPARATOR_DOUBLE -1.797693134862315708e+308 // -std::numeric_limits<double>::max()

double convertDecimalToDouble(in int64_t val, in uint64_t scale) {
  double rtn = double(val);
  if (val != NULL_DECIMAL) {
    rtn /= double(scale);
  }
  return rtn;
}

float unpack_pixel_coord_x(in int64_t xy) {
  return float(xy & 0x7FFF) * 0.25;
}

float unpack_pixel_coord_y(in int64_t xy) {
  return float((xy >> 16) & 0x7FFF) * 0.25;
}

double decompress_geo_coord_x(in int32_t x) {
  if (x == SEPARATOR_INT32) {
    return SEPARATOR_DOUBLE;
  }
  return (180.0 / double(0x7FFFFFFF)) * double(x);
}

double decompress_geo_coord_y(in int32_t y) {
  if (y == SEPARATOR_INT32) {
    return SEPARATOR_DOUBLE;
  }
  return (90.0 / double(0x7FFFFFFF)) * double(y);
}
