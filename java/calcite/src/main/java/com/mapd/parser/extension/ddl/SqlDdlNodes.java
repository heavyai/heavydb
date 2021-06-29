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

import com.mapd.parser.extension.ddl.omnisci.OmniSciOptionsMap;
import com.mapd.parser.extension.ddl.omnisci.OmniSciSqlDataTypeSpec;

import org.apache.calcite.schema.ColumnStrategy;
import org.apache.calcite.sql.SqlCollation;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Utilities concerning {@link SqlNode} for DDL.
 */
public class SqlDdlNodes {
  private SqlDdlNodes() {}

  /** Creates a CREATE TABLE. */
  public static SqlCreateTable createTable(SqlParserPos pos,
          boolean replace,
          boolean temporary,
          boolean ifNotExists,
          SqlIdentifier name,
          SqlNodeList columnList,
          OmniSciOptionsMap withOptions,
          SqlNode query) {
    return new SqlCreateTable(
            pos, replace, temporary, ifNotExists, name, columnList, withOptions, query);
  }

  /** Creates a CREATE VIEW. */
  public static SqlCreateView createView(SqlParserPos pos,
          boolean replace,
          boolean ifNotExists,
          SqlIdentifier name,
          SqlNodeList columnList,
          SqlNode query) {
    return new SqlCreateView(pos, replace, ifNotExists, name, columnList, query);
  }

  /** Creates a column declaration. */
  public static SqlNode column(SqlParserPos pos,
          SqlIdentifier name,
          OmniSciSqlDataTypeSpec dataType,
          SqlNode defaultValue,
          ColumnStrategy strategy) {
    return new SqlColumnDeclaration(pos, name, dataType, defaultValue, strategy);
  }

  /** Creates an attribute definition. */
  public static SqlNode attribute(SqlParserPos pos,
          SqlIdentifier name,
          SqlDataTypeSpec dataType,
          SqlNode expression,
          SqlCollation collation) {
    return new SqlAttributeDefinition(pos, name, dataType, expression, collation);
  }

  /** Creates a CHECK constraint. */
  public static SqlNode check(SqlParserPos pos, SqlIdentifier name, SqlNode expression) {
    return new SqlCheckConstraint(pos, name, expression);
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

  /** Creates a SHARD KEY constraint. */
  public static SqlKeyConstraint shard(SqlParserPos pos, SqlIdentifier name) {
    return SqlKeyConstraint.shard(pos, name);
  }

  /** Creates a SHARED DICTIONARY constraint. */
  public static SqlKeyConstraint sharedDict(
          SqlParserPos pos, SqlIdentifier columnName, SqlIdentifier referencesColumn) {
    return SqlKeyConstraint.sharedDict(pos, columnName, referencesColumn);
  }

  /** File type for CREATE FUNCTION. */
  public enum FileType { FILE, JAR, ARCHIVE }
}
