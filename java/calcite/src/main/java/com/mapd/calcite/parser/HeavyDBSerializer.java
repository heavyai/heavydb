/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.parser;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.mapd.parser.extension.ddl.DdlResponse;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.externalize.HeavyDBRelJsonWriter;

public class HeavyDBSerializer {
  private static final Gson gson;

  static {
    gson = new GsonBuilder().excludeFieldsWithoutExposeAnnotation().create();
  }

  public static String toString(final RelNode rel) {
    if (rel == null) {
      return null;
    }
    final HeavyDBRelJsonWriter planWriter = new HeavyDBRelJsonWriter();
    rel.explain(planWriter);
    return planWriter.asString();
  }

  public static String toJsonString(final JsonSerializableDdl jsonSerializableDdl) {
    final DdlResponse ddlResponse = new DdlResponse();
    ddlResponse.setPayload(jsonSerializableDdl);
    return gson.toJson(ddlResponse);
  }
}
