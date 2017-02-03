/*
 * Some cool MapD Header
 */
package com.mapd.calcite.parser;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Table;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.TranslatableTable;

public class MapDViewImpl extends MapDViewTable {

    MapDViewImpl(MapDCatalogReader catalogReader, String catalogName, String schemaName,
            String name, boolean stream, String viewSql) {
        super(catalogReader, catalogName, schemaName, name, stream, viewSql);
    }

    public String getViewSql() {
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
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
        return typeFactory.createStructType(rowType.getFieldList());
    }

    @Override
    public RelNode toRel(RelOptTable.ToRelContext context, RelOptTable relOptTable) {
        return context.expandView(relOptTable.getRowType(), viewSql, null, null).rel;
    }
}
