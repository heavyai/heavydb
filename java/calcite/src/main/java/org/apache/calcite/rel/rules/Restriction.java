/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.calcite.rel.rules;

import java.util.List;

public class Restriction {
  public Restriction(
          String rDatabase, String rTable, String rColumn, List<String> rValues) {
    this.rDatabase = rDatabase;
    this.rTable = rTable;
    this.rColumn = rColumn;
    this.rValues = rValues;
  }
  String getRestrictionDatabase() {
    return rDatabase;
  };
  String getRestrictionTable() {
    return rTable;
  };
  String getRestrictionColumn() {
    return rColumn;
  };
  List<String> getRestrictionValues() {
    return rValues;
  };
  private String rDatabase;
  private String rTable;
  private String rColumn;
  private List<String> rValues;
}
