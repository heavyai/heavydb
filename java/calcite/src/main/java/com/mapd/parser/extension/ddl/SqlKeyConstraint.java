/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file is a modified derivative of Apache Calcite's org.apache.calcite.sql.ddl.SqlKeyConstraint.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.ImmutableNullableList;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Parse tree for {@code UNIQUE}, {@code PRIMARY KEY} constraints.
 *
 * <p>
 * And {@code FOREIGN KEY}, when we support it.
 */
public class SqlKeyConstraint extends SqlCall {
  private static final SqlSpecialOperator UNIQUE =
          new SqlSpecialOperator("UNIQUE", SqlKind.UNIQUE);

  protected static final SqlSpecialOperator PRIMARY =
          new SqlSpecialOperator("PRIMARY KEY", SqlKind.PRIMARY_KEY);

  protected static final SqlSpecialOperator FOREIGN =
          new SqlSpecialOperator("FOREIGN KEY", SqlKind.FOREIGN_KEY);

  private final SqlIdentifier name;
  private final SqlNodeList columnList;
  private final SqlIdentifier referencesTable;
  private final SqlNodeList referencesColumnList;

  /** Creates a SqlKeyConstraint. */
  SqlKeyConstraint(SqlParserPos pos, SqlIdentifier name, SqlNodeList columnList) {
    this(pos, name, columnList, null);
  }

  /** Creates a SqlKeyConstraint between two (or more) columns */
  SqlKeyConstraint(SqlParserPos pos,
          SqlIdentifier name,
          SqlNodeList columnList,
          SqlIdentifier referencesCol) {
    this(pos,
            name,
            columnList,
            referencesCol != null && !referencesCol.isSimple()
                    ? referencesCol.getComponent(0)
                    : null,
            referencesCol == null ? null
                    : SqlNodeList.of(referencesCol.isSimple()
                                    ? referencesCol
                                    : referencesCol.getComponent(1)));
  }

  SqlKeyConstraint(SqlParserPos pos,
          SqlIdentifier name,
          SqlNodeList columnList,
          SqlIdentifier referencesTable,
          SqlNodeList referencesColumnList) {
    super(pos);
    this.name = name;
    this.columnList = columnList;
    this.referencesTable = referencesTable;
    this.referencesColumnList = referencesColumnList;
  }

  /** Creates a UNIQUE constraint. */
  public static SqlKeyConstraint unique(
          SqlParserPos pos, SqlIdentifier name, SqlNodeList columnList) {
    return new SqlKeyConstraint(pos, name, columnList);
  }

  /** Creates a PRIMARY KEY constraint. */
  public static SqlKeyConstraint primary(
          SqlParserPos pos, SqlIdentifier name, SqlNodeList columnList) {
    return new SqlKeyConstraint(pos, name, columnList) {
      @Override
      public SqlOperator getOperator() {
        return PRIMARY;
      }
    };
  }

  /** Creates a FOREIGN KEY constraint. */
  public static SqlKeyConstraint foreign(SqlParserPos pos,
          SqlIdentifier name,
          SqlNodeList columnList,
          SqlIdentifier referencesTable,
          SqlNodeList referencesColumnList) {
    return new SqlKeyConstraint(pos, name, columnList, referencesTable, referencesColumnList) {
      @Override
      public SqlOperator getOperator() {
        return FOREIGN;
      }
    };
  }

  /** Creates a SHARD KEY constraint */
  public static SqlKeyConstraint shard(SqlParserPos pos, SqlIdentifier colName) {
    SqlNodeList colList = SqlNodeList.of(colName);
    return new SqlKeyConstraint(pos, new SqlIdentifier("SHARD_KEY", pos), colList);
  }

  /** CReates a SHARED DICTIONARY constraint */
  public static SqlKeyConstraint sharedDict(
          SqlParserPos pos, SqlIdentifier colName, SqlIdentifier referencesCol) {
    SqlNodeList colList = SqlNodeList.of(colName);
    return new SqlKeyConstraint(
            pos, new SqlIdentifier("SHARED_DICT", pos), colList, referencesCol);
  }

  @Override
  public SqlOperator getOperator() {
    return UNIQUE;
  }

  @Override
  public List<SqlNode> getOperandList() {
    return ImmutableNullableList.of(name, columnList);
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    if (name != null) {
      writer.keyword("CONSTRAINT");
      name.unparse(writer, 0, 0);
    }
    writer.keyword(getOperator().getName()); // "UNIQUE" or "PRIMARY KEY"
    columnList.unparse(writer, 1, 1);
    if (referencesTable != null) {
      writer.keyword("REFERENCES");
      referencesTable.unparse(writer, 0, 0);
      if (referencesColumnList != null) {
        referencesColumnList.unparse(writer, 1, 1);
      }
    }
  }

  @Override
  public String toString() {
    EscapedStringJsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    jsonBuilder.put(map, "type", "SQL_COLUMN_CONSTRAINT");

    jsonBuilder.put(map, "name", this.name == null ? null : this.name.toString());
    jsonBuilder.put(map, "constraintType", getOperator().getName());

    List<String> colNamesList = new ArrayList<String>();
    for (int i = 0; i < columnList.size(); i++) {
      SqlNode colNode = columnList.get(i);
      colNamesList.add(colNode.toString());
    }
    jsonBuilder.put(map, "columns", colNamesList);

    Map<String, Object> referencesMap = jsonBuilder.map();
    if (referencesTable != null) {
      jsonBuilder.put(referencesMap, "table", referencesTable.toString());
    }
    if (referencesColumnList != null) {
      List<String> referenceColNamesList = new ArrayList<String>();
      for (int i = 0; i < referencesColumnList.size(); i++) {
        SqlNode colNode = referencesColumnList.get(i);
        referenceColNamesList.add(colNode.toString());
      }
      jsonBuilder.put(referencesMap, "columns", referenceColNamesList);
      if (referenceColNamesList.size() == 1) {
        jsonBuilder.put(referencesMap, "column", referenceColNamesList.get(0));
      }
    }
    jsonBuilder.put(map, "references", referencesMap);

    return jsonBuilder.toJsonString(map);
  }
}
