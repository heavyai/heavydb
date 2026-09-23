/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
}
