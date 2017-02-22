package com.mapd.calcite.parser;

import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.schema.TranslatableTable;

public abstract class MapDViewTable extends MapDTable implements TranslatableTable {

    public String viewSql;

    MapDViewTable(MapDCatalogReader catalogReader, String catalogName, String schemaName,
            String name, boolean stream, String viewSql) {
        super(catalogReader, catalogName, schemaName, name, stream);
        this.viewSql = viewSql;
    }

    @Override
    public RelNode toRel(RelOptTable.ToRelContext context, RelOptTable relOptTable) {
        return context.expandView(relOptTable.getRowType(), viewSql, null, null).rel;
    }
}
