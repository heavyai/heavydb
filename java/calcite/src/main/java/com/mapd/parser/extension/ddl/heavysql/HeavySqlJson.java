/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl.heavysql;

import com.mapd.parser.extension.ddl.JsonSerializableDdl;

public class HeavySqlJson implements JsonSerializableDdl {
  @Override
  public String toString() {
    return toJsonString();
  }
}
