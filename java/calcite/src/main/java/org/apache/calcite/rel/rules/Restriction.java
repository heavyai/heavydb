/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
