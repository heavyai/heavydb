/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.parser;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mapd.metadata.LinestringSqlType;
import com.mapd.metadata.PointSqlType;
import com.mapd.metadata.PolygonSqlType;

import org.apache.calcite.rel.RelReferentialConstraint;
import org.apache.calcite.rel.RelReferentialConstraintImpl;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.IntPair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

import ai.heavy.thrift.server.TColumnType;
import ai.heavy.thrift.server.TDatumType;
import ai.heavy.thrift.server.TTableDetails;
import ai.heavy.thrift.server.TTypeInfo;

public class HeavyDBTable implements Table {
  private static final AtomicLong VERSION_PROVIDER = new AtomicLong();
  private static final Map<String, Double> ROW_COUNT_ESTIMATES =
          new ConcurrentHashMap<String, Double>();
  private static final ThreadLocal<Boolean> OPTIMIZER_METADATA_ENABLED =
          ThreadLocal.withInitial(() -> false);
  private static final ThreadLocal<Boolean> TRUST_UNENFORCED_TABLE_CONSTRAINTS =
          ThreadLocal.withInitial(() -> false);

  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(HeavyDBTable.class);
  private final TTableDetails rowInfo;
  private final String schemaName;
  private final String tableName;
  private final long version = VERSION_PROVIDER.incrementAndGet();
  private final HashSet<String> systemColumnNames;
  private final List<JsonObject> constraintObjects;

  public long getVersion() {
    return version;
  }

  public Double getRowCountEstimate() {
    if (rowInfo.isSetNum_rows() && rowInfo.getNum_rows() >= 0) {
      return (double) rowInfo.getNum_rows();
    }
    return null;
  }

  public static Double getRowCountEstimate(List<String> qualifiedName) {
    if (!OPTIMIZER_METADATA_ENABLED.get()) {
      return null;
    }
    if (qualifiedName == null || qualifiedName.size() < 2) {
      return null;
    }
    return ROW_COUNT_ESTIMATES.get(rowCountKey(qualifiedName.get(qualifiedName.size() - 2),
            qualifiedName.get(qualifiedName.size() - 1)));
  }

  private static String rowCountKey(String schemaName, String tableName) {
    return (schemaName + "." + tableName).toUpperCase(Locale.ROOT);
  }

  static boolean setOptimizerMetadataEnabled(boolean enabled) {
    final boolean previous = OPTIMIZER_METADATA_ENABLED.get();
    OPTIMIZER_METADATA_ENABLED.set(enabled);
    return previous;
  }

  static boolean setTrustUnenforcedTableConstraints(boolean trusted) {
    final boolean previous = TRUST_UNENFORCED_TABLE_CONSTRAINTS.get();
    TRUST_UNENFORCED_TABLE_CONSTRAINTS.set(trusted);
    return previous;
  }

  public static void invalidateMetadata(String schemaName, String tableName) {
    if (schemaName == null) {
      return;
    }
    final String schemaPrefix = schemaName.toUpperCase(Locale.ROOT) + ".";
    if (tableName == null || tableName.isEmpty()) {
      ROW_COUNT_ESTIMATES.keySet().removeIf(key -> key.startsWith(schemaPrefix));
      return;
    }
    ROW_COUNT_ESTIMATES.remove(rowCountKey(schemaName, tableName));
  }

  public HeavyDBTable(TTableDetails ri) {
    this(ri, null, null);
  }

  public HeavyDBTable(TTableDetails ri, String schemaName, String tableName) {
    rowInfo = ri;
    this.schemaName = schemaName;
    this.tableName = tableName;
    final Double rowCountEstimate = getRowCountEstimate();
    if (schemaName != null && tableName != null && rowCountEstimate != null) {
      ROW_COUNT_ESTIMATES.put(rowCountKey(schemaName, tableName), rowCountEstimate);
    }
    systemColumnNames = rowInfo.row_desc.stream()
                                .filter(row_desc -> row_desc.is_system)
                                .map(row_desc -> row_desc.col_name)
                                .collect(Collectors.toCollection(HashSet::new));
    constraintObjects = parseConstraintObjects();
  }

  @Override
  public RelDataType getRowType(RelDataTypeFactory rdtf) {
    RelDataTypeFactory.Builder builder = rdtf.builder();
    for (TColumnType tct : rowInfo.getRow_desc()) {
      HEAVYDBLOGGER.debug("'" + tct.col_name + "'"
              + " \t" + tct.getCol_type().getEncoding() + " \t"
              + tct.getCol_type().getFieldValue(TTypeInfo._Fields.TYPE) + " \t"
              + tct.getCol_type().nullable + " \t" + tct.getCol_type().is_array + " \t"
              + tct.getCol_type().precision + " \t" + tct.getCol_type().scale);
      builder.add(tct.col_name, createType(tct, rdtf));
    }
    return builder.build();
  }

  @Override
  public Statistic getStatistic() {
    if (!OPTIMIZER_METADATA_ENABLED.get()) {
      return Statistics.UNKNOWN;
    }
    Double rowCount = getRowCountEstimate();

    List<ImmutableBitSet> keys = getKeyStatistics();
    List<RelReferentialConstraint> referentialConstraints =
            getReferentialConstraintStatistics();
    if (rowCount != null || !keys.isEmpty() || !referentialConstraints.isEmpty()) {
      return Statistics.of(rowCount, keys, referentialConstraints, Collections.emptyList());
    }
    return Statistics.UNKNOWN;
  }

  private List<JsonObject> getConstraintObjects() {
    return constraintObjects;
  }

  private List<JsonObject> parseConstraintObjects() {
    if (!rowInfo.isSetKey_metainfo() || rowInfo.getKey_metainfo() == null
            || rowInfo.getKey_metainfo().isEmpty()) {
      return Collections.emptyList();
    }

    final JsonElement rootElement;
    try {
      rootElement = JsonParser.parseString(rowInfo.getKey_metainfo());
    } catch (RuntimeException ex) {
      HEAVYDBLOGGER.warn("Ignoring malformed key metadata for table " + tableName, ex);
      return Collections.emptyList();
    }
    if (!rootElement.isJsonArray()) {
      HEAVYDBLOGGER.warn("Ignoring non-array key metadata for table " + tableName);
      return Collections.emptyList();
    }

    List<JsonObject> constraintObjects = new ArrayList<JsonObject>();
    JsonArray metainfo = rootElement.getAsJsonArray();
    for (JsonElement element : metainfo) {
      if (!element.isJsonObject()) {
        continue;
      }
      JsonObject object = element.getAsJsonObject();
      if (!object.has("type") || !object.get("type").isJsonPrimitive()) {
        continue;
      }
      try {
        String type = object.get("type").getAsString();
        final boolean isKey = "PRIMARY KEY".equalsIgnoreCase(type)
                || "UNIQUE".equalsIgnoreCase(type);
        final boolean isForeignKey = "FOREIGN KEY".equalsIgnoreCase(type);
        if ((!isKey && !isForeignKey) || !hasNonEmptyStringArray(object, "columns")) {
          continue;
        }
        if (object.has("enforced")
                && (!object.get("enforced").isJsonPrimitive()
                        || !object.get("enforced").getAsJsonPrimitive().isBoolean())) {
          continue;
        }
        if (isForeignKey
                && (!hasString(object, "foreign_table")
                        || !hasNonEmptyStringArray(object, "foreign_columns")
                        || !hasNonNegativeIntegerArray(
                                object, "foreign_column_ordinals")
                        || object.getAsJsonArray("columns").size()
                                != object.getAsJsonArray("foreign_columns").size()
                        || object.getAsJsonArray("columns").size()
                                != object.getAsJsonArray("foreign_column_ordinals")
                                           .size())) {
          continue;
        }
        constraintObjects.add(object);
      } catch (RuntimeException ex) {
        HEAVYDBLOGGER.warn(
                "Ignoring malformed table constraint metadata for table " + tableName,
                ex);
      }
    }
    return Collections.unmodifiableList(constraintObjects);
  }

  private static boolean hasString(JsonObject object, String fieldName) {
    return object.has(fieldName) && object.get(fieldName).isJsonPrimitive()
            && object.get(fieldName).getAsJsonPrimitive().isString()
            && !object.get(fieldName).getAsString().isEmpty();
  }

  private static boolean hasNonEmptyStringArray(JsonObject object, String fieldName) {
    if (!object.has(fieldName) || !object.get(fieldName).isJsonArray()
            || object.getAsJsonArray(fieldName).size() == 0) {
      return false;
    }
    HashSet<String> values = new HashSet<String>();
    for (JsonElement value : object.getAsJsonArray(fieldName)) {
      if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()
              || value.getAsString().isEmpty()) {
        return false;
      }
      if (!values.add(value.getAsString().toUpperCase(Locale.ROOT))) {
        return false;
      }
    }
    return true;
  }

  private static boolean hasNonNegativeIntegerArray(
          JsonObject object, String fieldName) {
    if (!object.has(fieldName) || !object.get(fieldName).isJsonArray()
            || object.getAsJsonArray(fieldName).size() == 0) {
      return false;
    }
    HashSet<Integer> values = new HashSet<Integer>();
    for (JsonElement value : object.getAsJsonArray(fieldName)) {
      if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
        return false;
      }
      try {
        final String encodedValue = value.getAsString();
        final int ordinal = Integer.parseInt(encodedValue);
        if (!encodedValue.matches("0|[1-9][0-9]*") || ordinal < 0
                || !values.add(ordinal)) {
          return false;
        }
      } catch (RuntimeException ex) {
        return false;
      }
    }
    return true;
  }

  private Map<String, Integer> getColumnOrdinals() {
    Map<String, Integer> columnOrdinals = new HashMap<String, Integer>();
    for (int i = 0; i < rowInfo.getRow_descSize(); ++i) {
      columnOrdinals.put(
              rowInfo.getRow_desc().get(i).getCol_name().toUpperCase(Locale.ROOT), i);
    }
    return columnOrdinals;
  }

  private List<Integer> getColumnOrdinals(
          JsonObject constraintObject, Map<String, Integer> columnOrdinals) {
    if (!constraintObject.has("columns") || !constraintObject.get("columns").isJsonArray()) {
      return Collections.emptyList();
    }

    List<Integer> ordinals = new ArrayList<Integer>();
    for (JsonElement columnElement : constraintObject.getAsJsonArray("columns")) {
      String columnName = columnElement.getAsString().toUpperCase(Locale.ROOT);
      Integer ordinal = columnOrdinals.get(columnName);
      if (ordinal == null) {
        return Collections.emptyList();
      }
      ordinals.add(ordinal);
    }
    return ordinals;
  }

  private List<String> getStringList(JsonObject object, String fieldName) {
    if (!object.has(fieldName) || !object.get(fieldName).isJsonArray()) {
      return Collections.emptyList();
    }
    List<String> values = new ArrayList<String>();
    for (JsonElement value : object.getAsJsonArray(fieldName)) {
      values.add(value.getAsString());
    }
    return values;
  }

  private List<Integer> getIntegerList(JsonObject object, String fieldName) {
    if (!object.has(fieldName) || !object.get(fieldName).isJsonArray()) {
      return Collections.emptyList();
    }
    List<Integer> values = new ArrayList<Integer>();
    for (JsonElement value : object.getAsJsonArray(fieldName)) {
      values.add(value.getAsInt());
    }
    return values;
  }

  private List<String> getQualifiedName(String localTableName) {
    if (schemaName == null || schemaName.isEmpty()) {
      return Collections.singletonList(localTableName);
    }
    List<String> qualifiedName = new ArrayList<String>();
    qualifiedName.add(schemaName);
    qualifiedName.add(localTableName);
    return qualifiedName;
  }

  private List<ImmutableBitSet> getKeyStatistics() {
    Map<String, Integer> columnOrdinals = getColumnOrdinals();
    List<ImmutableBitSet> keys = new ArrayList<ImmutableBitSet>();
    for (JsonObject constraintObject : getConstraintObjects()) {
      if (!isConstraintTrusted(constraintObject)) {
        continue;
      }
      String type = constraintObject.get("type").getAsString();
      if (!"PRIMARY KEY".equalsIgnoreCase(type) && !"UNIQUE".equalsIgnoreCase(type)) {
        continue;
      }
      List<Integer> ordinals = getColumnOrdinals(constraintObject, columnOrdinals);
      if (!ordinals.isEmpty() && keyColumnsAreNotNull(ordinals)) {
        keys.add(ImmutableBitSet.of(ordinals));
      }
    }
    return keys;
  }

  private boolean keyColumnsAreNotNull(List<Integer> ordinals) {
    for (int ordinal : ordinals) {
      if (ordinal < 0 || ordinal >= rowInfo.getRow_descSize() ||
              rowInfo.getRow_desc().get(ordinal).getCol_type().nullable) {
        return false;
      }
    }
    return true;
  }

  private boolean isConstraintTrusted(JsonObject constraintObject) {
    return constraintObject.has("enforced")
            && constraintObject.get("enforced").isJsonPrimitive()
            && constraintObject.get("enforced").getAsJsonPrimitive().isBoolean()
            && constraintObject.get("enforced").getAsBoolean()
            || TRUST_UNENFORCED_TABLE_CONSTRAINTS.get();
  }

  private List<RelReferentialConstraint> getReferentialConstraintStatistics() {
    if (tableName == null) {
      return Collections.emptyList();
    }

    Map<String, Integer> sourceColumnOrdinals = getColumnOrdinals();
    List<RelReferentialConstraint> referentialConstraints =
            new ArrayList<RelReferentialConstraint>();
    for (JsonObject constraintObject : getConstraintObjects()) {
      if (!isConstraintTrusted(constraintObject)) {
        continue;
      }
      String type = constraintObject.get("type").getAsString();
      if (!"FOREIGN KEY".equalsIgnoreCase(type)
              || !constraintObject.has("foreign_table")
              || !constraintObject.has("foreign_columns")) {
        continue;
      }

      String foreignTableName = constraintObject.get("foreign_table").getAsString();
      List<Integer> sourceOrdinals =
              getColumnOrdinals(constraintObject, sourceColumnOrdinals);
      List<String> foreignColumnNames = getStringList(constraintObject, "foreign_columns");
      List<Integer> foreignColumnOrdinals =
              getIntegerList(constraintObject, "foreign_column_ordinals");
      if (sourceOrdinals.isEmpty() || sourceOrdinals.size() != foreignColumnNames.size()
              || sourceOrdinals.size() != foreignColumnOrdinals.size()) {
        continue;
      }

      List<IntPair> columnPairs = new ArrayList<IntPair>();
      for (int i = 0; i < sourceOrdinals.size(); ++i) {
        columnPairs.add(IntPair.of(sourceOrdinals.get(i), foreignColumnOrdinals.get(i)));
      }
      referentialConstraints.add(RelReferentialConstraintImpl.of(
              getQualifiedName(tableName),
              getQualifiedName(foreignTableName),
              columnPairs));
    }
    return referentialConstraints;
  }

  @Override
  public Schema.TableType getJdbcTableType() {
    return Schema.TableType.TABLE;
  }

  private RelDataType createType(TColumnType value, RelDataTypeFactory typeFactory) {
    RelDataType cType = getRelDataType(value.col_type.type,
            value.col_type.precision,
            value.col_type.scale,
            typeFactory);

    if (value.col_type.is_array) {
      cType = typeFactory.createArrayType(
              typeFactory.createTypeWithNullability(cType, true), -1);
    }

    if (value.col_type.isNullable()) {
      return typeFactory.createTypeWithNullability(cType, true);
    } else {
      return cType;
    }
  }

  // Convert our TDataumn type in to a base calcite SqlType
  // todo confirm whether it is ok to ignore thinsg like lengths
  // since we do not use them on the validator side of the calcite 'fence'
  private RelDataType getRelDataType(
          TDatumType dType, int precision, int scale, RelDataTypeFactory typeFactory) {
    switch (dType) {
      case TINYINT:
        return typeFactory.createSqlType(SqlTypeName.TINYINT);
      case SMALLINT:
        return typeFactory.createSqlType(SqlTypeName.SMALLINT);
      case INT:
        return typeFactory.createSqlType(SqlTypeName.INTEGER);
      case BIGINT:
        return typeFactory.createSqlType(SqlTypeName.BIGINT);
      case FLOAT:
        return typeFactory.createSqlType(SqlTypeName.FLOAT);
      case DECIMAL:
        return typeFactory.createSqlType(SqlTypeName.DECIMAL, precision, scale);
      case DOUBLE:
        return typeFactory.createSqlType(SqlTypeName.DOUBLE);
      case STR:
        return typeFactory.createSqlType(SqlTypeName.VARCHAR, 50);
      case TIME:
        return typeFactory.createSqlType(SqlTypeName.TIME);
      case TIMESTAMP:
        return typeFactory.createSqlType(SqlTypeName.TIMESTAMP, precision);
      case DATE:
        return typeFactory.createSqlType(SqlTypeName.DATE);
      case BOOL:
        return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
      case INTERVAL_DAY_TIME:
        return typeFactory.createSqlType(SqlTypeName.INTERVAL_DAY);
      case INTERVAL_YEAR_MONTH:
        return typeFactory.createSqlType(SqlTypeName.INTERVAL_YEAR_MONTH);
      case POINT:
        return typeFactory.createSqlType(SqlTypeName.GEOMETRY);
      // return new PointSqlType();
      case MULTIPOINT:
        return typeFactory.createSqlType(SqlTypeName.GEOMETRY);
      // return new MultipointSqlType();
      case LINESTRING:
        return typeFactory.createSqlType(SqlTypeName.GEOMETRY);
      // return new LinestringSqlType();
      case MULTILINESTRING:
        return typeFactory.createSqlType(SqlTypeName.GEOMETRY);
      // return new MultilinestringSqlType();
      case POLYGON:
        return typeFactory.createSqlType(SqlTypeName.GEOMETRY);
      // return new PolygonSqlType();
      case MULTIPOLYGON:
        return typeFactory.createSqlType(SqlTypeName.GEOMETRY);
      // return new MultipolygonSqlType();
      default:
        throw new AssertionError(dType.name());
    }
  }

  @Override
  public boolean isRolledUp(String string) {
    // will set to false by default
    return false;
  }

  @Override
  public boolean rolledUpColumnValidInsideAgg(
          String string, SqlCall sc, SqlNode sn, CalciteConnectionConfig ccc) {
    throw new UnsupportedOperationException(
            "rolledUpColumnValidInsideAgg Not supported yet.");
  }

  public boolean isSystemColumn(final String columnName) {
    return systemColumnNames.contains(columnName);
  }
}
