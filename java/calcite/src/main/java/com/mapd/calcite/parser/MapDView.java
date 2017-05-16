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

import com.mapd.thrift.server.TTableDetails;
import org.apache.calcite.plan.RelOptTable;
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

public class MapDView extends MapDTable implements TranslatableTable {

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDView.class);
  final MapDParser parser;
  private final String viewSql;

  public MapDView(String view_sql, TTableDetails ri, MapDParser mp) {
    super(ri);
    this.viewSql = view_sql;
    parser = mp;
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
      return parser.queryToSqlNode(viewSql, true).rel;
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
      final RelRoot relAlg = parser.queryToSqlNode(viewSql, true);
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
