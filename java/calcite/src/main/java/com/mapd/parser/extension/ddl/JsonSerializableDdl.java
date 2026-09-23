/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl;

import com.mapd.calcite.parser.HeavyDBSerializer;

public interface JsonSerializableDdl {
  default String toJsonString() {
    return HeavyDBSerializer.toJsonString(this);
  }
}
