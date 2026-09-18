/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlCreate;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.SqlWriterConfig;
import org.apache.calcite.sql.dialect.CalciteSqlDialect;
import org.apache.calcite.sql.parser.Span;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.ImmutableNullableList;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Parse tree for {@code CREATE ROLE} statement.
 */
public class SqlCreateRole extends SqlCreate {
  public final SqlIdentifier role;

  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("CREATE_ROLE", SqlKind.OTHER_DDL);

  /** Creates a SqlCreateRole. */
  SqlCreateRole(SqlParserPos pos, SqlIdentifier role) {
    super(OPERATOR, pos, false, false);
    this.role = Objects.requireNonNull(role);
  }

  public List<SqlNode> getOperandList() {
    return ImmutableNullableList.of(role);
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.keyword("CREATE");
    writer.keyword("ROLE");
    role.unparse(writer, leftPrec, rightPrec);
  }

  @Override
  public String toString() {
    EscapedStringJsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();
    jsonBuilder.put(map, "role", this.role.toString());
    map.put("command", "CREATE_ROLE");
    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);
    return jsonBuilder.toJsonString(payload);
  }
}
