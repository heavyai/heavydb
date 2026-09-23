/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl;

import java.util.HashMap;
import java.util.Map;

public abstract class SqlOptionsBuilder {
  protected Map<String, String> options;

  public void addOption(final String attribute, final String value) {
    if (options == null) {
      options = new HashMap<>();
    }

    options.put(attribute, sanitizeOptionValue(value));
  }

  private String sanitizeOptionValue(final String value) {
    String sanitizedValue = value;
    if (value.startsWith("'") && value.endsWith("'")) {
      sanitizedValue = value.substring(1, value.length() - 1);
    }
    return sanitizedValue;
  }
}
