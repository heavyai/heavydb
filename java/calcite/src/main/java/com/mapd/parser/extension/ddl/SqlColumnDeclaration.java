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

import com.google.common.collect.ImmutableList;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBGeoTypeNameSpec;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBSqlDataTypeSpec;
import com.mapd.parser.extension.ddl.heavydb.HeavyDBTypeNameSpec;

import org.apache.calcite.schema.ColumnStrategy;
import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlTypeNameSpec;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.EscapedStringJsonBuilder;

import java.util.List;
import java.util.Map;

/**
 * Parse tree for {@code UNIQUE}, {@code PRIMARY KEY} constraints.
 *
 * <p>
 * And {@code FOREIGN KEY}, when we support it.
 */
public class SqlColumnDeclaration extends SqlCall {
  private static final SqlSpecialOperator OPERATOR =
          new SqlSpecialOperator("COLUMN_DECL", SqlKind.COLUMN_DECL);

  public final SqlIdentifier name;
  public final HeavyDBSqlDataTypeSpec dataType;
  public final SqlNode defaultValue;
  public final ColumnStrategy strategy;

  /** Creates a SqlColumnDeclaration; use {@link SqlDdlNodes#column}. */
  SqlColumnDeclaration(SqlParserPos pos,
          SqlIdentifier name,
          HeavyDBSqlDataTypeSpec dataType,
          SqlNode defaultValue,
          ColumnStrategy strategy) {
    super(pos);
    this.name = name;
    this.dataType = dataType;
    this.defaultValue = defaultValue;
    this.strategy = strategy;
  }

  @Override
  public SqlOperator getOperator() {
    return OPERATOR;
  }

  @Override
  public List<SqlNode> getOperandList() {
    return ImmutableList.of(name, dataType);
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    name.unparse(writer, 0, 0);
    dataType.unparse(writer, 0, 0);
    if (dataType.getNullable() != null && !dataType.getNullable()) {
      writer.keyword("NOT NULL");
    }
    if (defaultValue != null) {
      switch (strategy) {
        case DEFAULT:
          writer.keyword("DEFAULT");
          exp(writer);
          break;
        default:
          throw new AssertionError("unexpected: " + strategy);
      }
    }
  }

  private void exp(SqlWriter writer) {
    if (writer.isAlwaysUseParentheses()) {
      defaultValue.unparse(writer, 0, 0);
    } else {
      writer.sep("(");
      defaultValue.unparse(writer, 0, 0);
      writer.sep(")");
    }
  }

  @Override
  public String toString() {
    EscapedStringJsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    jsonBuilder.put(map, "type", "SQL_COLUMN_DECLARATION");

    jsonBuilder.put(map, "name", name == null ? null : name.toString());

    jsonBuilder.put(
            map, "default", defaultValue == null ? null : defaultValue.toString());
    jsonBuilder.put(map, "nullable", !Boolean.FALSE.equals(dataType.getNullable()));
    jsonBuilder.put(map, "encodingType", dataType.getEncodingString());
    jsonBuilder.put(map, "encodingSize", dataType.getEncodingSize());

    SqlTypeNameSpec dataTypeSpec = dataType.getTypeNameSpec();
    if (dataTypeSpec instanceof HeavyDBGeoTypeNameSpec) {
      map = ((HeavyDBGeoTypeNameSpec) dataTypeSpec).toJsonMap(map);
    } else {
      boolean isText = false;
      if (dataTypeSpec instanceof HeavyDBTypeNameSpec) {
        HeavyDBTypeNameSpec heavyDBDataTypeSpec = (HeavyDBTypeNameSpec) dataTypeSpec;
        if (heavyDBDataTypeSpec.getIsArray()) {
          jsonBuilder.put(map, "arraySize", heavyDBDataTypeSpec.getArraySize());
        }
        isText = heavyDBDataTypeSpec.getIsText();
      }
      jsonBuilder.put(
              map, "precision", ((SqlBasicTypeNameSpec) dataTypeSpec).getPrecision());
      jsonBuilder.put(map, "scale", ((SqlBasicTypeNameSpec) dataTypeSpec).getScale());
      if (isText) {
        jsonBuilder.put(map, "sqltype", "TEXT");
      } else {
        jsonBuilder.put(map,
                "sqltype",
                dataType == null ? null : dataTypeSpec.getTypeName().toString());
      }
    }
    return jsonBuilder.toJsonString(map);
  }
}
