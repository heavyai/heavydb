/*
 * Copyright 2019 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "sqltypes.h"

namespace DatumFetcher {

template <typename T>
T getDatumVal(const Datum& d);

template <>
int8_t getDatumVal(const Datum& d) {
  return d.tinyintval;
}

template <>
int16_t getDatumVal(const Datum& d) {
  return d.smallintval;
}

template <>
int32_t getDatumVal(const Datum& d) {
  return d.intval;
}

template <>
int64_t getDatumVal(const Datum& d) {
  return d.bigintval;
}

template <>
uint8_t getDatumVal(const Datum& d) {
  return d.tinyintval;
}

template <>
uint16_t getDatumVal(const Datum& d) {
  return d.smallintval;
}

template <>
float getDatumVal(const Datum& d) {
  return d.floatval;
}

template <>
double getDatumVal(const Datum& d) {
  return d.doubleval;
}

}  // namespace DatumFetcher
