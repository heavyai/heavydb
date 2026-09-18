/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

public class DdlResponse {
  @Expose
  private final String statementType = "DDL";
  @Expose
  private JsonSerializableDdl payload;

  public void setPayload(final JsonSerializableDdl payload) {
    this.payload = payload;
  }
}
