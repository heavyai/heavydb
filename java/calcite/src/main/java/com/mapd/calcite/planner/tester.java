/*
 * Copyright 2022 HEAVY.AI, Inc.
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

package com.mapd.calcite.planner;

import com.mapd.calcite.parser.HeavyDBParser;
import com.mapd.calcite.parser.HeavyDBParserOptions;
import com.mapd.calcite.parser.HeavyDBSchema;
import com.mapd.calcite.parser.HeavyDBSerializer;
import com.mapd.calcite.parser.HeavyDBSqlOperatorTable;
import com.mapd.calcite.parser.HeavyDBUser;

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.validate.SqlConformanceEnum;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.slf4j.LoggerFactory;

import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

public class tester {
  final static org.slf4j.Logger HEAVYDBLOGGER = LoggerFactory.getLogger(tester.class);

  public static void main(String[] args) {
    final SqlStdOperatorTable stdOpTab = SqlStdOperatorTable.instance();

    HeavyDBUser mdu = new HeavyDBUser("admin", "passwd", "omnisci", -1, null);
    HeavyDBSchema dbSchema =
            new HeavyDBSchema("<<PATH_TO_DATA_DIR>>", null, -1, mdu, null, null);
    final SchemaPlus rootSchema = Frameworks.createRootSchema(true);
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(rootSchema.add("omnisci", dbSchema))
                    .operatorTable(stdOpTab)
                    .parserConfig(SqlParser.configBuilder()
                                          .setConformance(SqlConformanceEnum.LENIENT)
                                          .setUnquotedCasing(Casing.UNCHANGED)
                                          .setCaseSensitive(false)
                                          .build())
                    .build();

    Planner p = Frameworks.getPlanner(config);

    SqlNode parseR = null;
    try {
      parseR = p.parse("<<QUERY>>");
    } catch (SqlParseException ex) {
      Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
    }

    SqlNode validateR = null;
    try {
      p.validate(parseR);
    } catch (ValidationException ex) {
      Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
    }
    RelRoot relR = null;
    try {
      relR = p.rel(validateR);
    } catch (RelConversionException ex) {
      Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
    }
    HEAVYDBLOGGER.error("Result was " + relR);
    HEAVYDBLOGGER.error("Result project() " + relR.project());
    HEAVYDBLOGGER.error("Result project() " + RelOptUtil.toString(relR.project()));
    HEAVYDBLOGGER.error("Json Version \n" + HeavyDBSerializer.toString(relR.project()));

    // now do with HeavyDB parser
    Supplier<HeavyDBSqlOperatorTable> operatorTable =
            new Supplier<HeavyDBSqlOperatorTable>() {
              @Override
              public HeavyDBSqlOperatorTable get() {
                return new HeavyDBSqlOperatorTable(SqlStdOperatorTable.instance());
              }
            };
    HeavyDBParser mp = new HeavyDBParser("<<PATH_TO_DATA_DIR>>", operatorTable, -1, null);
    mp.setUser(mdu);

    try {
      HeavyDBParserOptions mdpo = new HeavyDBParserOptions();
      HEAVYDBLOGGER.error("HeavyDBParser result: \n" + mp.processSql("<<QUERY>>", mdpo));
    } catch (SqlParseException ex) {
      Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
    } catch (ValidationException ex) {
      Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
    } catch (RelConversionException ex) {
      Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  // /** User-defined aggregate function. */
  // public static class MyCountAggFunction extends SqlAggFunction {
  // public MyCountAggFunction() {
  // super("MY_COUNT", null, SqlKind.OTHER_FUNCTION, ReturnTypes.BIGINT, null,
  // OperandTypes.ANY, SqlFunctionCategory.NUMERIC, false, false);
  // }
  //
  // @SuppressWarnings("deprecation")
  // public List<RelDataType> getParameterTypes(RelDataTypeFactory typeFactory) {
  // return ImmutableList.of(typeFactory.createSqlType(SqlTypeName.ANY));
  // }
  //
  // @SuppressWarnings("deprecation")
  // public RelDataType getReturnType(RelDataTypeFactory typeFactory) {
  // return typeFactory.createSqlType(SqlTypeName.BIGINT);
  // }
  //
  // public RelDataType deriveType(SqlValidator validator,
  // SqlValidatorScope scope, SqlCall call) {
  // // Check for COUNT(*) function. If it is we don't
  // // want to try and derive the "*"
  // if (call.isCountStar()) {
  // return validator.getTypeFactory().createSqlType(SqlTypeName.BIGINT);
  // }
  // return super.deriveType(validator, scope, call);
  // }
  // }
}
