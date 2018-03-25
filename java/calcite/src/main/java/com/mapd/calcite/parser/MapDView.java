/*
 * Copyright 2017 MapD Technologies, Inc.
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
package com.mapd.calcite.parser;

import static com.mapd.calcite.parser.MapDParser.CURRENT_PARSER;
import java.util.ArrayList;

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

import com.mapd.thrift.server.TTableDetails;

public class MapDView extends MapDTable implements TranslatableTable {

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDView.class);
  private final String viewSql;
  private SqlIdentifierCapturer accessObjects;

  public MapDView(String view_sql, TTableDetails ri, MapDParser mp) {
    super(ri);
    this.viewSql = view_sql;
    try {
      accessObjects = mp.captureIdentifiers(view_sql, true);
    } catch (SqlParseException e) {
      MAPDLOGGER.error("error parsing view SQL: " + view_sql, e);
      accessObjects = new SqlIdentifierCapturer();
    }
  }
  
  public SqlIdentifierCapturer getAccessedObjects() {
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
    try {
      return CURRENT_PARSER.get().queryToSqlNode(viewSql, new ArrayList<>(), true).rel;
    } catch (SqlParseException ex) {
      assert false;
      return null;
    } catch (ValidationException ex) {
      assert false;
      return null;
    } catch (RelConversionException ex) {
      assert false;
      return null;
    }
  }

  @Override
  public RelDataType getRowType(RelDataTypeFactory rdtf) {
    try {
      final RelRoot relAlg = CURRENT_PARSER.get().queryToSqlNode(viewSql, new ArrayList<>(), true);
      return relAlg.validatedRowType;
    } catch (SqlParseException e) {
      assert false;
      return null;
    } catch (ValidationException ex) {
      assert false;
      return null;
    } catch (RelConversionException ex) {
      assert false;
      return null;
    }
  }
}
