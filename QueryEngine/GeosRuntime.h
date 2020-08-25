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

#ifndef QUERYENGINE_RUNTIMEFUNCTIONSGEOS_H
#define QUERYENGINE_RUNTIMEFUNCTIONSGEOS_H

extern "C" bool Geos_Wkb_Wkb(int op,
                             int arg1_type,
                             int8_t* arg1_coords,
                             int64_t arg1_coords_size,
                             int32_t* arg1_meta1,
                             int64_t arg1_meta1_size,
                             int32_t* arg1_meta2,
                             int64_t arg1_meta2_size,
                             int32_t arg1_ic,
                             int32_t arg1_srid,
                             int arg2_type,
                             int8_t* arg2_coords,
                             int64_t arg2_coords_size,
                             int32_t* arg2_meta1,
                             int64_t arg2_meta1_size,
                             int32_t* arg2_meta2,
                             int64_t arg2_meta2_size,
                             int32_t arg2_ic,
                             int32_t arg2_srid,
                             int* result_type,
                             int8_t** result_coords,
                             int64_t* result_coords_size,
                             int32_t** result_meta1,
                             int64_t* result_meta1_size,
                             int32_t** result_meta2,
                             int64_t* result_meta2_size);

extern "C" bool Geos_Wkb_double(int op,
                                int arg1_type,
                                int8_t* arg1_coords,
                                int64_t arg1_coords_size,
                                int32_t* arg1_meta1,
                                int64_t arg1_meta1_size,
                                int32_t* arg1_meta2,
                                int64_t arg1_meta2_size,
                                int32_t arg1_ic,
                                int32_t arg1_srid,
                                double arg2,
                                int* result_type,
                                int8_t** result_coords,
                                int64_t* result_coords_size,
                                int32_t** result_meta1,
                                int64_t* result_meta1_size,
                                int32_t** result_meta2,
                                int64_t* result_meta2_size);

extern "C" bool Geos_Wkb(int op,
                         int arg_type,
                         int8_t* arg_coords,
                         int64_t arg_coords_size,
                         int32_t* arg_meta1,
                         int64_t arg_meta1_size,
                         int32_t* arg_meta2,
                         int64_t arg_meta2_size,
                         int32_t arg_ic,
                         int32_t arg_srid,
                         bool* result);

#endif  // QUERYENGINE_RUNTIMEFUNCTIONSGEOS_H
