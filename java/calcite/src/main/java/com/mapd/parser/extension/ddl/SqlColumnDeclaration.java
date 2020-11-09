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
import com.mapd.parser.extension.ddl.omnisci.OmniSciGeoTypeNameSpec;
import com.mapd.parser.extension.ddl.omnisci.OmniSciSqlDataTypeSpec;
import com.mapd.parser.extension.ddl.omnisci.OmniSciTypeNameSpec;

import org.apache.calcite.avatica.SqlType;
import org.apache.calcite.schema.ColumnStrategy;
import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlTypeNameSpec;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;
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
  public final OmniSciSqlDataTypeSpec dataType;
  public final SqlNode expression;
  public final ColumnStrategy strategy;

  /** Creates a SqlColumnDeclaration; use {@link SqlDdlNodes#column}. */
  SqlColumnDeclaration(SqlParserPos pos,
          SqlIdentifier name,
          OmniSciSqlDataTypeSpec dataType,
          SqlNode expression,
          ColumnStrategy strategy) {
    super(pos);
    this.name = name;
    this.dataType = dataType;
    this.expression = expression;
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
    if (expression != null) {
      switch (strategy) {
        case VIRTUAL:
        case STORED:
          writer.keyword("AS");
          exp(writer);
          writer.keyword(strategy.name());
          break;
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
      expression.unparse(writer, 0, 0);
    } else {
      writer.sep("(");
      expression.unparse(writer, 0, 0);
      writer.sep(")");
    }
  }

  @Override
  public String toString() {
    EscapedStringJsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    jsonBuilder.put(map, "type", "SQL_COLUMN_DECLARATION");

    jsonBuilder.put(map, "name", this.name == null ? null : this.name.toString());

    jsonBuilder.put(map,
            "expression",
            this.expression == null ? null : this.expression.toString());
    jsonBuilder.put(map, "nullable", this.strategy == ColumnStrategy.NULLABLE);
    jsonBuilder.put(map, "encodingType", this.dataType.getEncodingString());
    jsonBuilder.put(map, "encodingSize", this.dataType.getEncodingSize());

    SqlTypeNameSpec dataTypeSpec = this.dataType.getTypeNameSpec();
    if (dataTypeSpec instanceof OmniSciGeoTypeNameSpec) {
      map = ((OmniSciGeoTypeNameSpec) dataTypeSpec).toJsonMap(map);
    } else {
      boolean isText = false;
      if (dataTypeSpec instanceof OmniSciTypeNameSpec) {
        OmniSciTypeNameSpec omniSciDataTypeSpec = (OmniSciTypeNameSpec) dataTypeSpec;
        if (omniSciDataTypeSpec.getIsArray()) {
          jsonBuilder.put(map, "arraySize", omniSciDataTypeSpec.getArraySize());
        }
        isText = omniSciDataTypeSpec.getIsText();
      }
      jsonBuilder.put(
              map, "precision", ((SqlBasicTypeNameSpec) dataTypeSpec).getPrecision());
      jsonBuilder.put(map, "scale", ((SqlBasicTypeNameSpec) dataTypeSpec).getScale());
      if (isText) {
        jsonBuilder.put(map, "sqltype", "TEXT");
      } else {
        jsonBuilder.put(map,
                "sqltype",
                this.dataType == null ? null : dataTypeSpec.getTypeName().toString());
      }
    }
    return jsonBuilder.toJsonString(map);
  }
}
