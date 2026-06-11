/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TableFunctionsTesting.h"

/*
  This file contains testing compile-time UDTFs that arguments use
  FlatBuffer storage and that implementations use the NestedArray API
  only.
 */

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_copy__generic_cpu_template(TableFunctionManager& mgr,
                                                        const Column<T>& inputs,
                                                        Column<T>& outputs) {
  auto size = inputs.size();
  mgr.set_output_item_values_total_number(0, inputs.getNofValues());
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (inputs.isNull(i)) {
      outputs.setNull(i);
    } else {
      outputs[i] = inputs[i];
    }
  }
  return size;
}

#define INSTANTIATE_CT_COPY(T)                                      \
  template NEVER_INLINE HOST int32_t ct_copy__generic_cpu_template( \
      TableFunctionManager& mgr, const Column<T>& inputs, Column<T>& outputs);

INSTANTIATE_CT_COPY(TextEncodingNone)
INSTANTIATE_CT_COPY(GeoLineString)
INSTANTIATE_CT_COPY(GeoPolygon)
INSTANTIATE_CT_COPY(GeoMultiPoint)
INSTANTIATE_CT_COPY(GeoMultiLineString)
INSTANTIATE_CT_COPY(GeoMultiPolygon)

template <typename T>
NEVER_INLINE HOST int32_t ct_concat__generic_cpu_template(TableFunctionManager& mgr,
                                                          const Column<T>& input1,
                                                          const Column<T>& input2,
                                                          Column<T>& outputs) {
  auto size = input1.size();
  mgr.set_output_item_values_total_number(0,
                                          input1.getNofValues() + input2.getNofValues());
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (input1.isNull(i)) {
      if (input2.isNull(i)) {
        outputs.setNull(i);
      } else {
        outputs[i] = input2[i];
      }
    } else if (input2.isNull(i)) {
      outputs[i] = input1[i];
    } else {
      outputs[i] = input1[i];
      outputs.concatItem(i, input2[i]);
    }
  }
  return size;
}

#define INSTANTIATE_CT_CONCAT(T)                                      \
  template NEVER_INLINE HOST int32_t ct_concat__generic_cpu_template( \
      TableFunctionManager& mgr,                                      \
      const Column<T>& input1,                                        \
      const Column<T>& input2,                                        \
      Column<T>& outputs);

INSTANTIATE_CT_CONCAT(TextEncodingNone)

template <typename T>
NEVER_INLINE HOST int32_t ct_concat__generic2_cpu_template(TableFunctionManager& mgr,
                                                           const Column<T>& input1,
                                                           const T& input2,
                                                           Column<T>& output) {
  auto size = input1.size();
  mgr.set_output_item_values_total_number(0,
                                          input1.getNofValues() + size * input2.size());
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (input1.isNull(i) || input2.isNull()) {
      output.setNull(i);
    } else {
      output[i] = input1[i];
      output[i] += input2;
    }
  }
  return size;
}

#define INSTANTIATE_CT_CONCAT2(T)                                      \
  template NEVER_INLINE HOST int32_t ct_concat__generic2_cpu_template( \
      TableFunctionManager& mgr,                                       \
      const Column<T>& input1,                                         \
      const T& input2,                                                 \
      Column<T>& outputs);

INSTANTIATE_CT_CONCAT2(TextEncodingNone)
INSTANTIATE_CT_CONCAT2(Array<double>)

template <typename T>
NEVER_INLINE HOST int32_t ct_concat__generic3_cpu_template(TableFunctionManager& mgr,
                                                           const T& input1,
                                                           const Column<T>& input2,
                                                           Column<T>& output) {
  auto size = input2.size();
  mgr.set_output_item_values_total_number(0,
                                          input2.getNofValues() + size * input1.size());
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; i++) {
    if (input2.isNull(i) || input1.isNull()) {
      output.setNull(i);
    } else {
      output[i] = input1;
      output[i] += input2[i];
    }
  }
  return size;
}

#define INSTANTIATE_CT_CONCAT3(T)                                      \
  template NEVER_INLINE HOST int32_t ct_concat__generic3_cpu_template( \
      TableFunctionManager& mgr,                                       \
      const T& input1,                                                 \
      const Column<T>& input2,                                         \
      Column<T>& outputs);

INSTANTIATE_CT_CONCAT3(TextEncodingNone)

#endif
