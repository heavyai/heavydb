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
 */

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlDdl;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.JsonBuilder;

import java.util.List;
import java.util.Map;

/**
 * Class that encapsulates all information associated with a ALTER TABLE DDL
 * command.
 */
public class SqlAlterTable extends SqlDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("ALTER_TABLE", SqlKind.OTHER_DDL);

  /**
   * ALTER TABLE DDL syntax variants:
   *
   * SET OPTIONS:
   *    ALTER TABLE <table> [ SET (<option> = <value> [, ... ] ) ]
   * RENAME TABLE
   *    ALTER TABLE <table> RENAME TO <new_table>
   * RENAME COLUMN:
   *    ALTER TABLE <table> RENAME <column> to <new_column>
   * ADD COLUMN:
   *    ALTER TABLE <table>
   *          ADD [COLUMN] <column> <type> [NOT NULL] [ENCODING <encodingSpec>];
   *    ALTER TABLE <table>
   *          ADD (<column> <type> [NOT NULL] [ENCODING <encodingSpec>], ...);
   *    ALTER TABLE <table> ADD (<column> <type> DEFAULT <value>);
   * DROP COLUMN:
   *    ALTER TABLE <table> DROP COLUMN <column_1>[, <column_2>, ...];
   */

  public enum AlterType {
    RENAME_TABLE,
    RENAME_COLUMN,
    ADD_COLUMN,
    DROP_COLUMN,
    ALTER_OPTIONS
  }

  public static class Builder extends SqlOptionsBuilder {
    private SqlParserPos pos;
    private AlterType alterType;
    private String tableName;
    private String newTableName;
    private String columnName;
    private String newColumnName;
    private SqlNodeList columnList;

    public void setPos(final SqlParserPos pos) {
      this.pos = pos;
    }

    public void setTableName(final String tableName) {
      this.tableName = tableName;
    }

    public void alterOptions() {
      this.alterType = AlterType.ALTER_OPTIONS;
      // Options should be read in directly to base class
    }

    public void alterTableName(final String newTableName) {
      this.alterType = AlterType.RENAME_TABLE;
      this.newTableName = newTableName;
    }

    public void alterColumnName(final String columnName, final String newColumnName) {
      this.alterType = AlterType.RENAME_COLUMN;
      this.columnName = columnName;
      this.newColumnName = newColumnName;
    }

    public void addColumnList(final SqlNodeList columnList) {
      this.alterType = AlterType.ADD_COLUMN;
      this.columnList = columnList;
    }

    public void dropColumn(final SqlNodeList columnList) {
      this.alterType = AlterType.DROP_COLUMN;
      this.columnList = columnList;
    }

    public SqlAlterTable build() {
      return new SqlAlterTable(pos,
              alterType,
              tableName,
              newTableName,
              columnName,
              newColumnName,
              columnList,
              super.options);
    }
  }

  @Expose
  private AlterType alterType;
  @Expose
  private String tableName;
  @Expose
  private String newTableName;
  @Expose
  private String columnName;
  @Expose
  private String newColumnName;
  @Expose
  private String command;
  @Expose
  private SqlNodeList columnList;
  @Expose
  private Map<String, String> options;

  public SqlAlterTable(final SqlParserPos pos, final SqlIdentifier name) {
    super(OPERATOR, pos);
    this.tableName = name.toString();
  }

  public SqlAlterTable(final SqlParserPos pos,
          final AlterType alterType,
          final String tableName,
          final String newTableName,
          final String columnName,
          final String newColumnName,
          final SqlNodeList columnList,
          final Map<String, String> options) {
    super(OPERATOR, pos);
    this.alterType = alterType;
    this.tableName = tableName;
    this.newTableName = newTableName;
    this.columnName = columnName;
    this.newColumnName = newColumnName;
    this.options = options;
    this.columnList = columnList;
    this.command = OPERATOR.getName();
  }

  // @Override
  public List<SqlNode> getOperandList() {
    // Add the operands here
    return null;
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    writer.keyword("ALTER");
    writer.keyword("TABLE");
    // add other options data here when/as necessary
  }

  @Override
  public String toString() {
    JsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    map.put("command", "ALTER_TABLE");
    map.put("tableName", this.tableName.toString());
    switch (this.alterType) {
      case RENAME_TABLE:
        map.put("alterType", "RENAME_TABLE");
        map.put("newTableName", this.newTableName.toString());
        break;
      case RENAME_COLUMN:
        map.put("alterType", "RENAME_COLUMN");
        map.put("columnName", this.columnName.toString());
        map.put("newColumnName", this.newColumnName.toString());
        break;
      case ADD_COLUMN:
        map.put("alterType", "ADD_COLUMN");
        if (this.columnList != null) {
          List<Object> elements_list = jsonBuilder.list();
          for (SqlNode elementNode : this.columnList) {
            if (!(elementNode instanceof SqlCall)) {
              throw new CalciteException("Column definition for table "
                              + this.columnName.toString()
                              + " is invalid: " + elementNode.toString(),
                      null);
            }
            elements_list.add(elementNode);
          }
          map.put("columnData", elements_list);
        }

        break;
      case DROP_COLUMN:
        map.put("alterType", "DROP_COLUMN");
        map.put("columnData", this.columnList.toString());
        break;
      case ALTER_OPTIONS:
        map.put("alterType", "ALTER_OPTIONS");
        map.put("options", this.options);
        break;
    }

    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);
    return jsonBuilder.toJsonString(payload);
  }
}
