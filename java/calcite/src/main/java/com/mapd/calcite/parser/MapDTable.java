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
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.plan.RelOptSchema;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelCollations;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.logical.LogicalTableScan;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.schema.TranslatableTable;
import org.apache.calcite.sql.SqlAccessType;
import org.apache.calcite.sql.validate.SqlModality;
import org.apache.calcite.sql.validate.SqlMonotonicity;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;

/**
 *
 * @author michael
 */
/**
 * MapD implementation of {@link org.apache.calcite.prepare.Prepare.PreparingTable}.
 */
public class MapDTable implements Prepare.PreparingTable {

  protected final MapDCatalogReader catalogReader;
  private final boolean stream;
  private final List<Map.Entry<String, RelDataType>> columnList
          = Lists.newArrayList();
  private RelDataType rowType;
  private List<RelCollation> collationList;
  protected final List<String> names;
  private final Set<String> monotonicColumnSet = Sets.newHashSet();
  private double rowCount = 0l;

  public MapDTable(MapDCatalogReader catalogReader, String catalogName,
          String schemaName, String name, boolean stream) {
    this.catalogReader = catalogReader;
    this.stream = stream;
    this.names = ImmutableList.of(catalogName, schemaName, name);
  }

  public static MapDTable create(MapDCatalogReader catalogReader,
          MapDDatabase schema, String name, boolean stream) {
    MapDTable table
            = new MapDTable(catalogReader, schema.getCatalogName(), schema.getSchemaName(),
                    name, stream);
    schema.addTable(name);
    return table;
  }

//    public <T> T unwrap(Class<T> clazz) {
//      if (clazz.isInstance(this)) {
//        return clazz.cast(this);
//      }
//      if (clazz.isAssignableFrom(Table.class)) {
//        return clazz.cast(
//            new JdbcTest.AbstractModifiableTable(Util.last(names)) {
//              @Override public RelDataType
//              getRowType(RelDataTypeFactory typeFactory) {
//                return typeFactory.createStructType(rowType.getFieldList());
//              }
//
//              @Override public Collection getModifiableCollection() {
//                return null;
//              }
//
//              @Override public <E> Queryable<E>
//              asQueryable(QueryProvider queryProvider, SchemaPlus schema,
//                  String tableName) {
//                return null;
//              }
//
//              @Override public Type getElementType() {
//                return null;
//              }
//
//              @Override public Expression getExpression(SchemaPlus schema,
//                  String tableName, Class clazz) {
//                return null;
//              }
//            });
//      }
//      return null;
//    }
  @Override
  public double getRowCount() {
    return rowCount;
  }

  @Override
  public RelOptSchema getRelOptSchema() {
    return catalogReader;
  }

  @Override
  public RelNode toRel(RelOptTable.ToRelContext context) {
    if (this instanceof TranslatableTable) {
      return ((TranslatableTable) this).toRel(context, this);
    } else {
      return LogicalTableScan.create(context.getCluster(), this);
    }
  }

  @Override
  public List<RelCollation> getCollationList() {
    return collationList;
  }

  @Override
  public RelDistribution getDistribution() {
    return RelDistributions.BROADCAST_DISTRIBUTED;
  }

  @Override
  public boolean isKey(ImmutableBitSet columns) {
    return false;
  }

  @Override
  public RelDataType getRowType() {
    return rowType;
  }

  @Override
  public boolean supportsModality(SqlModality modality) {
    return modality == (stream ? SqlModality.STREAM : SqlModality.RELATION);
  }

  public void onRegister(RelDataTypeFactory typeFactory) {
    rowType = typeFactory.createStructType(columnList);
    collationList = deduceMonotonicity(this);
  }

    private static List<RelCollation> deduceMonotonicity(
          Prepare.PreparingTable table) {
    final List<RelCollation> collationList = Lists.newArrayList();

    // Deduce which fields the table is sorted on.
    int i = -1;
    for (RelDataTypeField field : table.getRowType().getFieldList()) {
      ++i;
      final SqlMonotonicity monotonicity
              = table.getMonotonicity(field.getName());
      if (monotonicity != SqlMonotonicity.NOT_MONOTONIC) {
        final RelFieldCollation.Direction direction
                = monotonicity.isDecreasing()
                        ? RelFieldCollation.Direction.DESCENDING
                        : RelFieldCollation.Direction.ASCENDING;
        collationList.add(
                RelCollations.of(
                        new RelFieldCollation(i, direction,
                                RelFieldCollation.NullDirection.UNSPECIFIED)));
      }
    }
    return collationList;

  }

  @Override
  public List<String> getQualifiedName() {
    return names;
  }

  @Override
  public SqlMonotonicity getMonotonicity(String columnName) {
    return monotonicColumnSet.contains(columnName)
            ? SqlMonotonicity.INCREASING
            : SqlMonotonicity.NOT_MONOTONIC;
  }

  @Override
  public SqlAccessType getAllowedAccess() {
    return SqlAccessType.ALL;
  }

  @Override
  public Expression getExpression(Class clazz) {
    throw new UnsupportedOperationException();
  }

  public void addColumn(String name, RelDataType type) {
    columnList.add(Pair.of(name, type));
  }

  public void addMonotonic(String name) {
    monotonicColumnSet.add(name);
    assert Pair.left(columnList).contains(name);
  }

  @Override
  public RelOptTable extend(List<RelDataTypeField> extendedFields) {
    final MapDTable table = new MapDTable(catalogReader, names.get(0),
            names.get(1), names.get(2), stream);
    table.columnList.addAll(columnList);
    table.columnList.addAll(extendedFields);
    table.onRegister(catalogReader.typeFactory);
    return table;
  }

  @Override
  public <T> T unwrap(Class<T> clazz) {
    if (clazz.isInstance(this)) {
      return clazz.cast(this);
    }
    return null;
  }

  void setRowCount(double rows) {
    rowCount = rows;
  }
}
