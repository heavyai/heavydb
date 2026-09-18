/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/QueryEngine.h"

CUstream getQueryEngineCudaStreamForDevice(
    int device_num) {  // NOTE: CUstream is cudaStream_t
  return QueryEngine::getInstance()->getCudaStreamForDevice(device_num);
}
