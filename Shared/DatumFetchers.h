/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sqltypes.h"

namespace DatumFetcher {

template <typename T>
inline T getDatumVal(const Datum& d);

template <>
inline int8_t getDatumVal(const Datum& d) {
  return d.tinyintval;
}

template <>
inline int16_t getDatumVal(const Datum& d) {
  return d.smallintval;
}

template <>
inline int32_t getDatumVal(const Datum& d) {
  return d.intval;
}

template <>
inline int64_t getDatumVal(const Datum& d) {
  return d.bigintval;
}

template <>
inline uint8_t getDatumVal(const Datum& d) {
  return d.tinyintval;
}

template <>
inline uint16_t getDatumVal(const Datum& d) {
  return d.smallintval;
}

template <>
inline float getDatumVal(const Datum& d) {
  return d.floatval;
}

template <>
inline double getDatumVal(const Datum& d) {
  return d.doubleval;
}

}  // namespace DatumFetcher
