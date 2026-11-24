/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.parser;

import static com.mapd.calcite.parser.HeavyDBParser.CURRENT_PARSER;

import com.mapd.metadata.MetaConnect.ViewSources;

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

import ai.heavy.thrift.server.TTableDetails;

public class HeavyDBView extends HeavyDBTable implements TranslatableTable {
  private final String viewSql;
  private final Object initLock = new Object();
  private static final class Expansion {
    final SqlIdentifierCapturer accessObjects;
    final RelRoot root;
    final ViewSources sources;

    Expansion(SqlIdentifierCapturer accessObjects, RelRoot root, ViewSources sources) {
      this.accessObjects = accessObjects;
      this.root = root;
      this.sources = sources;
    }
  }
  private volatile Expansion expansion;

  public HeavyDBView(String view_sql, TTableDetails ri, HeavyDBParser mp) {
    this(view_sql, ri, mp, null, null);
  }

  public HeavyDBView(String view_sql,
          TTableDetails ri,
          HeavyDBParser mp,
          String schemaName,
          String tableName) {
    super(ri, schemaName, tableName);
    this.viewSql = view_sql;
    // Calcite 1.41 asks for view row types while the parser context is already
    // established. Defer planning until that point instead of recursively
    // planning the view from catalog-object construction.
  }

  public String toString() {
    Expansion plan = expansion;
    return "View SQL: " + viewSql + "\n"
            + "Accessed Objects\n" + (plan == null ? null : plan.accessObjects);
  }

  public SqlIdentifierCapturer getAccessedObjects() {
    Expansion plan = expansion;
    return (plan == null ? getExpansion() : plan).accessObjects;
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
    return getExpansion().root.rel;
  }

  @Override
  public RelDataType getRowType(RelDataTypeFactory rdtf) {
    return getExpansion().root.validatedRowType;
  }

  private Expansion getExpansion() {
    Expansion plan = expansion;
    if (plan != null && plan.sources.isCurrent()) return plan;
    synchronized (initLock) {
      plan = expansion;
      if (plan != null && plan.sources.isCurrent()) return plan;

      HeavyDBParser parser = CURRENT_PARSER.get();
      if (parser == null) {
        throw new IllegalStateException(
                "HeavyDBView initialization requires a parser context.");
      }

      try {
        HeavyDBParserOptions parserOptions = new HeavyDBParserOptions();
        SqlIdentifierCapturer accessObjects = plan == null
                ? parser.captureIdentifiers(viewSql, parserOptions.isLegacySyntax())
                : plan.accessObjects;
        ViewSources sources =
                new ViewSources(parser.resolveSelectIdentifiers(accessObjects));
        expansion = new Expansion(
                accessObjects, parser.queryToRelNode(viewSql, parserOptions), sources);
        return expansion;
      } catch (SqlParseException | ValidationException | RelConversionException e) {
        throw new IllegalStateException(
                "Could not plan view SQL: " + viewSql + ": " + e.getMessage(), e);
      }
    }
  }
}
