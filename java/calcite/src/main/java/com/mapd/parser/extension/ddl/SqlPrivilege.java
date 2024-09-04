/*
 * Copyright 2023 HEAVY.AI, Inc.
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

package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.JsonBuilder;

import java.util.List;
import java.util.Map;

/**
 * A <code>SqlPrivilege</code> represents a privilege type along with any
 * additional parameters pertinent to the privilege. For example, columns may
 * be specified for certain privileges and would be represented as additional
 * parameters.
 *
 * This class extends {@link SqlNodeList} as a convenience.
 */
public class SqlPrivilege extends SqlNodeList {
  private String type;
  private List<SqlIdentifier> columnTargets;

  SqlPrivilege(String type, List<SqlIdentifier> columnTargets, SqlParserPos pos) {
    super(pos);
    this.type = type;
    this.columnTargets = columnTargets;
  }

  Map<String, Object> toJson(JsonBuilder jsonBuilder) {
    Map<String, Object> map = jsonBuilder.map();
    map.put("type", type);
    if (columnTargets != null && !columnTargets.isEmpty()) {
      List<Object> columns = jsonBuilder.list();
      for (SqlIdentifier column : columnTargets) {
        columns.add(column.toString());
      }
      map.put("columns", columns);
    }
    return map;
  }
}
