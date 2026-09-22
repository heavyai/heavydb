/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.rel.rules;

import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;

final class HeavyDBAggregateCallUtils {
  private HeavyDBAggregateCallUtils() {}

  static boolean hasExtendedOperands(AggregateCall call) {
    return call.distinctKeys != null || !call.rexList.isEmpty();
  }

  static boolean isDeterministic(AggregateCall call) {
    if (!call.getAggregation().isDeterministic()) {
      return false;
    }
    for (RexNode expression : call.rexList) {
      if (!RexUtil.isDeterministic(expression)) {
        return false;
      }
    }
    return true;
  }
}
