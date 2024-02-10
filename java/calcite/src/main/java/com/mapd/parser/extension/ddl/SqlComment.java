/*
 * Copyright 2024 HEAVY.AI, Inc.
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

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that represents COMMENT ON command:
 *
 *   COMMENT ON (TABLE | COLUMN)  <object_name> IS (<string_literal> | NULL);
 */
public class SqlComment extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("COMMENT", SqlKind.OTHER_DDL);

  /**
   */
  public enum TargetObjectType { TABLE, COLUMN }

  public static class Builder extends SqlOptionsBuilder {
    private SqlParserPos pos;
    private TargetObjectType targetObjectType;
    private String tableName;
    private String columnName;
    private boolean setToNull = false;
    private String comment;

    public void setPos(final SqlParserPos pos) {
      this.pos = pos;
    }

    public void setColumnType() {
      this.targetObjectType = TargetObjectType.COLUMN;
    }

    public void setTableType() {
      this.targetObjectType = TargetObjectType.TABLE;
    }

    public void setTableName(final String tableName) {
      this.tableName = tableName;
    }

    public void setColumnName(final String columnName) {
      this.columnName = columnName;
    }

    public void setComment(final String comment) {
      this.comment = comment;
    }

    public void setToNull() {
      this.setToNull = true;
    }

    public SqlComment build() {
      return new SqlComment(
              pos, targetObjectType, tableName, columnName, comment, setToNull);
    }
  }

  @Expose
  private TargetObjectType targetObjectType;
  @Expose
  private String tableName;
  @Expose
  private String columnName;
  @Expose
  private String comment;
  @Expose
  private boolean setToNull;

  public SqlComment(final SqlParserPos pos,
          final TargetObjectType targetObjectType,
          final String tableName,
          final String columnName,
          final String comment,
          final boolean setToNull) {
    super(OPERATOR, pos);
    this.targetObjectType = targetObjectType;
    this.tableName = tableName;
    this.columnName = columnName;
    this.comment = comment;
    this.setToNull = setToNull;
  }
}
