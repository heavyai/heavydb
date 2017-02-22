package com.mapd.calcite.parser;

import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.sql.parser.SqlParseException;

public class MapDViewImpl extends MapDViewTable {

    MapDViewImpl(MapDCatalogReader catalogReader, String catalogName, String schemaName,
            String name, boolean stream, String viewSql, final MapDParser parser) {
        super(catalogReader, catalogName, schemaName, name, stream, viewSql);
        this.parser = parser;
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

    private final MapDParser parser;
}
