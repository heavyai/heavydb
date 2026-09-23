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
 * This file is a modified derivative of Apache Calcite's org.apache.calcite.sql.ddl.SqlCreateTable.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.SqlWriterConfig;
import org.apache.calcite.sql.dialect.CalciteSqlDialect;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.JsonBuilder;

import java.util.List;
import java.util.Map;

/**
 * Class that encapsulates all information associated with a INSERT INTO TABLE DDL
 * command.
 */
public class SqlInsertIntoTable extends SqlDdl {
  public final SqlNode name;
  public final SqlNodeList columnList;
  public SqlNode query = null;

  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("INSERT_INTO_TABLE_AS_SELECT", SqlKind.OTHER_DDL);

  public SqlInsertIntoTable(
          SqlParserPos pos, SqlNode table, SqlNode query, SqlNodeList columnList) {
    super(OPERATOR, pos);
    this.name = table;
    this.query = query;
    this.columnList = columnList; // may be null
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.keyword("INSERT");
    writer.keyword("INTO");
    name.unparse(writer, leftPrec, rightPrec);
    if (columnList != null) {
      SqlWriter.Frame frame = writer.startList("(", ")");
      for (SqlNode c : columnList) {
        writer.sep(",");
        c.unparse(writer, 0, 0);
      }
      writer.endList(frame);
    }
    if (query != null) {
      writer.newlineAndIndent();
      query.unparse(writer, 0, 0);
    }
  }

  @Override
  public String toString() {
    JsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    map.put("command", "INSERT_INTO_TABLE_AS_SELECT");
    map.put("name", this.name.toString());

    // By default ... toString() seems to single-quote too much stuff
    //    for the SELECT stmt to be executed later
    //    ->
    //    use PrettyWriter to output a cleaner SQL statement
    //
    SqlWriterConfig c = SqlPrettyWriter.config()
                                .withDialect(CalciteSqlDialect.DEFAULT)
                                .withQuoteAllIdentifiers(false)
                                .withSelectListItemsOnSeparateLines(false)
                                .withWhereListItemsOnSeparateLines(false)
                                .withValuesListNewline(false);
    SqlPrettyWriter writer = new SqlPrettyWriter(c);
    this.query.unparse(writer, 0, 0);
    map.put("query", writer.toString());

    if (columnList != null) {
      List<Object> col_list = jsonBuilder.list();
      for (SqlNode col : this.columnList) {
        col_list.add(col.toString());
      }
      jsonBuilder.put(map, "columns", col_list);
    }

    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);

    return jsonBuilder.toJsonString(payload);
  }
}
