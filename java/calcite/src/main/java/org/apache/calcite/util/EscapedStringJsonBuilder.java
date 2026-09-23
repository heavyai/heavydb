/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.calcite.util;

import org.apache.calcite.util.JsonBuilder;
import org.apache.commons.text.StringEscapeUtils;

import java.util.List;
import java.util.Map;

public class EscapedStringJsonBuilder extends JsonBuilder {
  @Override
  public void append(StringBuilder buf, int indent, Object o) {
    if (o instanceof String) {
      buf.append('"').append(StringEscapeUtils.escapeJson((String) o)).append('"');
    } else if ((o == null) || (o instanceof Map) || (o instanceof List)
            || (o instanceof Number) || (o instanceof Boolean)) {
      super.append(buf, indent, o);
    } else {
      buf.append(o);
    }
  }
}
