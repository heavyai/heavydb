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
import org.apache.calcite.schema.Table;
import org.apache.calcite.schema.TranslatableTable;

public abstract class MapDViewTable extends MapDTable implements TranslatableTable {
    public String viewSql;

    MapDViewTable(MapDCatalogReader catalogReader, String catalogName, String schemaName,
            String name, boolean stream, String viewSql) {
        super(catalogReader, catalogName, schemaName, name, stream);
        this.viewSql = viewSql;
    }

    public RelNode toRel(RelOptTable.ToRelContext context, RelOptTable relOptTable) {
        return context.expandView(relOptTable.getRowType(), viewSql, null, null).rel;
    }
}

