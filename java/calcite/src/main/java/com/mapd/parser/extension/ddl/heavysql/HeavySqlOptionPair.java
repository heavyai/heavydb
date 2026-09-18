/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl.heavysql;

import com.mapd.parser.extension.ddl.heavysql.HeavySqlSanitizedString;

import org.apache.calcite.util.Pair;

public class HeavySqlOptionPair extends Pair<String, HeavySqlSanitizedString> {
  public HeavySqlOptionPair(String option, HeavySqlSanitizedString value) {
    super(option, value);
  }
}
