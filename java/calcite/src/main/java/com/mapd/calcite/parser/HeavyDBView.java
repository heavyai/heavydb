/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.parser;

import static com.mapd.calcite.parser.HeavyDBParser.CURRENT_PARSER;

import com.mapd.calcite.parser.HeavyDBParserOptions;

import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.TranslatableTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

import ai.heavy.thrift.server.TTableDetails;

public class HeavyDBView extends HeavyDBTable implements TranslatableTable {
  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(HeavyDBView.class);
  private final String viewSql;
  private volatile SqlIdentifierCapturer accessObjects;
  private volatile RelRoot viewRelRoot;
  private volatile boolean initialized = false;
  private final Object initLock = new Object();

  public HeavyDBView(String view_sql, TTableDetails ri, HeavyDBParser mp) {
    super(ri);
    this.viewSql = view_sql;
    // Calcite 1.41 asks for view row types while the parser context is already
    // established. Defer planning until that point instead of recursively
    // planning the view from catalog-object construction.
  }

  public String toString() {
    return "View SQL: " + viewSql + "\n"
            + "Accessed Objects\n" + accessObjects;
  }

  public SqlIdentifierCapturer getAccessedObjects() {
    ensureInitialized();
    return accessObjects;
  }

  String getViewSql() {
    return viewSql;
  }

  @Override
  public Schema.TableType getJdbcTableType() {
    return Schema.TableType.VIEW;
  }

  @Override
  public RelNode toRel(RelOptTable.ToRelContext context, RelOptTable relOptTable) {
    ensureInitialized();
    return viewRelRoot.rel;
  }

  @Override
  public RelDataType getRowType(RelDataTypeFactory rdtf) {
    ensureInitialized();
    return viewRelRoot.validatedRowType;
  }

  private void ensureInitialized() {
    if (initialized) {
      return;
    }

    synchronized (initLock) {
      if (initialized) {
        return;
      }

      HeavyDBParser parser = CURRENT_PARSER.get();
      if (parser == null) {
        throw new IllegalStateException(
                "HeavyDBView initialization requires a parser context.");
      }

      try {
        HeavyDBParserOptions parserOptions = new HeavyDBParserOptions();
        viewRelRoot = parser.queryToRelNode(viewSql, parserOptions);
        accessObjects =
                parser.captureIdentifiers(viewSql, parserOptions.isLegacySyntax());
      } catch (SqlParseException e) {
        HEAVYDBLOGGER.error("error parsing view SQL: " + viewSql, e);
      } catch (ValidationException ex) {
        HEAVYDBLOGGER.error("error validating view SQL: " + viewSql, ex);
      } catch (RelConversionException ex) {
        HEAVYDBLOGGER.error("error doing Rel Conversion view SQL: " + viewSql, ex);
      } finally {
        initialized = true;
      }
    }
  }
}
