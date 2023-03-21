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

import com.mapd.parser.extension.ddl.heavydb.HeavyDBOptionsMap;

import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.sql.SqlCall;
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
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.ImmutableNullableList;
import org.apache.calcite.util.JsonBuilder;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Parse tree for {@code CREATE MODEL} statement.
 */
public class SqlCreateModel extends SqlCreate {
  public final SqlIdentifier modelType;
  public final SqlIdentifier name;
  public final SqlNode query;
  private final HeavyDBOptionsMap options;

  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("CREATE MODEL", SqlKind.OTHER_DDL);

  /** Creates a SqlCreateModel. */
  protected SqlCreateModel(SqlParserPos pos,
          boolean replace,
          boolean ifNotExists,
          SqlIdentifier modelType,
          SqlIdentifier name,
          HeavyDBOptionsMap withOptions,
          SqlNode query) {
    super(OPERATOR, pos, replace, ifNotExists);
    this.modelType = Objects.requireNonNull(modelType);
    this.name = Objects.requireNonNull(name);
    this.options = withOptions;
    this.query = query; // for "CREATE TABLE ... AS query"; may be null
  }

  public List<SqlNode> getOperandList() {
    return ImmutableNullableList.of(modelType, name, query);
  }

  @Override
  public void unparse(SqlWriter writer, int leftPrec, int rightPrec) {
    if (getReplace()) {
      writer.keyword("CREATE OR REPLACE");
    } else {
      writer.keyword("CREATE");
    }
    writer.keyword("MODEL");
    if (ifNotExists) {
      writer.keyword("IF NOT EXISTS");
    }
    writer.keyword("OF TYPE");
    modelType.unparse(writer, leftPrec, rightPrec);
    name.unparse(writer, leftPrec, rightPrec);
    if (query != null) {
      writer.keyword("AS");
      writer.newlineAndIndent();
      query.unparse(writer, 0, 0);
    }
  }

  @Override
  public String toString() {
    JsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    Map<String, Object> map = jsonBuilder.map();

    jsonBuilder.put(map, "command", "CREATE_MODEL");
    jsonBuilder.put(map, "type", this.modelType.toString());
    jsonBuilder.put(map, "name", this.name.toString());

    if (query != null) {
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
      jsonBuilder.put(map, "query", writer.toString());
    }

    jsonBuilder.put(map, "replace", getReplace());
    jsonBuilder.put(map, "ifNotExists", this.ifNotExists);

    map.put("options", this.options);

    Map<String, Object> payload = jsonBuilder.map();
    payload.put("payload", map);

    // To Debug:
    // System.out.println(jsonBuilder.toJsonString(payload))

    return jsonBuilder.toJsonString(payload);
  }
}
