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

import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.TranslatableTable;
import org.apache.calcite.sql.parser.SqlParseException;

class MapDView extends MapDTable implements TranslatableTable {

    MapDView(MapDCatalogReader catalogReader, String catalogName, String schemaName,
            String name, boolean stream, String viewSql, final MapDParser parser) {
        super(catalogReader, catalogName, schemaName, name, stream);
        this.viewSql = viewSql;
        this.parser = parser;
    }

    String getViewSql() {
        return viewSql;
    }

    @Override
    public Schema.TableType getJdbcTableType() {
        return Schema.TableType.VIEW;
    }

    @Override
    public Statistic getStatistic() {
        return Statistics.UNKNOWN;
    }

    @Override
    public RelDataType getRowType() {
        try {
            final RelRoot relAlg = parser.queryToSqlNode(viewSql, true);
            return relAlg.validatedRowType;
        } catch (SqlParseException e) {
            assert false;
            return null;
        }
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
        return getRowType();
    }

    @Override
    public RelNode toRel(RelOptTable.ToRelContext context, RelOptTable relOptTable) {
        return context.expandView(relOptTable.getRowType(), viewSql, null, null).rel;
    }

    private final String viewSql;
    private final MapDParser parser;
}
