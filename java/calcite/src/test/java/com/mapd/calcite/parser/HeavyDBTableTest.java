/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.mapd.calcite.parser;

import static org.junit.Assert.*;

import com.mapd.metadata.MetaConnect;

import org.apache.calcite.rel.RelReferentialConstraint;
import org.apache.calcite.schema.Table;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.IntPair;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import ai.heavy.thrift.server.TColumnType;
import ai.heavy.thrift.server.TDatumType;
import ai.heavy.thrift.server.TTableDetails;
import ai.heavy.thrift.server.TTypeInfo;

public class HeavyDBTableTest {
  private boolean previousEnabled;
  private boolean previousTrust;

  @Before
  public void setup() {
    previousEnabled = HeavyDBTable.setOptimizerMetadataEnabled(true);
    previousTrust = HeavyDBTable.setTrustUnenforcedTableConstraints(true);
  }

  @After
  public void teardown() {
    HeavyDBTable.setOptimizerMetadataEnabled(previousEnabled);
    HeavyDBTable.setTrustUnenforcedTableConstraints(previousTrust);
    HeavyDBTable.invalidateMetadata("constraint_test", "");
  }

  private TTableDetails details(String metadata, boolean nullable) {
    TColumnType column = new TColumnType();
    column.setCol_name("id");
    TTypeInfo type = new TTypeInfo();
    type.setType(TDatumType.INT);
    type.setNullable(nullable);
    column.setCol_type(type);
    TTableDetails details = new TTableDetails();
    details.setRow_desc(Collections.singletonList(column));
    details.setKey_metainfo(metadata);
    details.setNum_rows(10);
    return details;
  }

  private HeavyDBTable table(TTableDetails details) {
    return new HeavyDBTable(details, "constraint_test", "child");
  }

  private String unique() {
    return "[{\"type\":\"UNIQUE\",\"name\":\"uk\",\"columns\":[\"id\"],\"enforced\":"
            + "false}]";
  }

  private String foreignKey() {
    return "[{\"type\":\"FOREIGN KEY\",\"name\":\"fk\",\"columns\":[\"id\"],"
            + "\"foreign_table\":\"parent\",\"foreign_columns\":[\"id\"],\"enforced\":"
            + "false,"
            + "\"foreign_column_ordinals\":[99]}]";
  }

  @Test
  public void flagsOffExposeNoOptimizerMetadata() {
    TTableDetails details = details(unique(), false);
    details.setNum_rows(12);
    HeavyDBTable table = table(details);
    HeavyDBTable.setOptimizerMetadataEnabled(false);
    assertNull(table.getStatistic().getRowCount());
    assertFalse(table.getStatistic().isKey(ImmutableBitSet.of(0)));
    assertNull(
            HeavyDBTable.getRowCountEstimate(Arrays.asList("constraint_test", "child")));
  }

  @Test
  public void keysRequireTrustAndNonNullColumns() {
    HeavyDBTable table = table(details(unique(), false));
    HeavyDBTable.setTrustUnenforcedTableConstraints(false);
    assertFalse(table.getStatistic().isKey(ImmutableBitSet.of(0)));
    HeavyDBTable.setTrustUnenforcedTableConstraints(true);
    assertTrue(table.getStatistic().isKey(ImmutableBitSet.of(0)));
    assertFalse(
            table(details(unique(), true)).getStatistic().isKey(ImmutableBitSet.of(0)));
  }

  @Test
  public void mixedKeysPreserveCompositeOrdinalsAndRejectNullableColumns() {
    TTableDetails details = details("["
                    + "{\"type\":\"PRIMARY KEY\",\"columns\":[\"id\"],\"enforced\":false},"
                    + "{\"type\":\"UNIQUE\",\"columns\":[\"nullable_code\"],\"enforced\":false},"
                    + "{\"type\":\"UNIQUE\",\"columns\":[\"ID\",\"code\"],\"enforced\":false}]",
            false);
    TColumnType id = details.getRow_desc().get(0);
    TColumnType nullable = id.deepCopy();
    nullable.setCol_name("nullable_code");
    nullable.getCol_type().setNullable(true);
    TColumnType code = id.deepCopy();
    code.setCol_name("code");
    details.setRow_desc(Arrays.asList(id, nullable, code));
    assertEquals(Arrays.asList(ImmutableBitSet.of(0), ImmutableBitSet.of(0, 2)),
            table(details).getStatistic().getKeys());
  }

  @Test
  public void foreignKeysUseRuntimeOrdinalsNotPersistedPositions() {
    TTableDetails details = details(foreignKey(), false);
    details.setForeign_key_column_ordinals(
            Collections.singletonMap("fk", Collections.singletonList(2)));
    assertEquals(Collections.singletonList(IntPair.of(0, 2)),
            table(details)
                    .getStatistic()
                    .getReferentialConstraints()
                    .get(0)
                    .getColumnPairs());
    assertTrue(table(details).referencesTable("PARENT"));
    assertFalse(table(details).referencesTable("unrelated"));
  }

  @Test
  public void compositeForeignKeysPreserveDeclaredPairOrder() {
    TTableDetails details = details("[{\"type\":\"FOREIGN KEY\",\"name\":\"fk\","
                    + "\"columns\":[\"other\",\"id\"],\"foreign_table\":\"parent\","
                    + "\"foreign_columns\":[\"b\",\"a\"],\"enforced\":false}]",
            false);
    TColumnType other = details.getRow_desc().get(0).deepCopy();
    other.setCol_name("other");
    details.setRow_desc(Arrays.asList(details.getRow_desc().get(0), other));
    details.setForeign_key_column_ordinals(
            Collections.singletonMap("fk", Arrays.asList(3, 1)));
    List<RelReferentialConstraint> constraints =
            table(details).getStatistic().getReferentialConstraints();
    assertEquals(1, constraints.size());
    RelReferentialConstraint constraint = constraints.get(0);
    assertEquals(Arrays.asList("constraint_test", "child"),
            constraint.getSourceQualifiedName());
    assertEquals(Arrays.asList("constraint_test", "parent"),
            constraint.getTargetQualifiedName());
    assertEquals(Arrays.asList(IntPair.of(1, 3), IntPair.of(0, 1)),
            constraint.getColumnPairs());
  }

  @Test
  public void missingOrInvalidRuntimeOrdinalsDisableForeignKeyMetadata() {
    TTableDetails details = details(foreignKey(), false);
    assertTrue(table(details).getStatistic().getReferentialConstraints().isEmpty());
    for (java.util.List<Integer> ordinals : Arrays.asList(Collections.singletonList(-1),
                 Arrays.asList(0, 1),
                 Collections.<Integer>emptyList())) {
      details.setForeign_key_column_ordinals(Collections.singletonMap("fk", ordinals));
      assertTrue(table(details).getStatistic().getReferentialConstraints().isEmpty());
    }
  }

  @Test
  public void foreignKeysRequireTrust() {
    TTableDetails details = details(foreignKey(), false);
    details.setForeign_key_column_ordinals(
            Collections.singletonMap("fk", Collections.singletonList(0)));
    HeavyDBTable.setTrustUnenforcedTableConstraints(false);
    assertTrue(table(details).getStatistic().getReferentialConstraints().isEmpty());
  }

  @Test
  public void malformedAndLegacyMetadataAreIgnored() {
    for (String metadata : Arrays.asList("",
                 "[]",
                 "not json",
                 "{}",
                 "{\"type\":\"PRIMARY KEY\"}",
                 "[{\"type\":\"SHARD KEY\",\"name\":\"id\"}]",
                 "[{\"type\":\"SHARD KEY\",\"name\":\"id\"},"
                         + "{\"type\":\"SHARED DICTIONARY\",\"name\":\"id\","
                         + "\"foreign_table\":\"t\",\"foreign_column\":\"id\"}]",
                 "[{\"type\":\"PRIMARY KEY\",\"columns\":[]}]",
                 "[{\"type\":\"UNIQUE\",\"columns\":[\"id\",\"ID\"]}]",
                 "[{\"type\":\"UNIQUE\",\"columns\":[\"absent\"]}]",
                 "[{\"type\":\"FOREIGN KEY\",\"name\":\"fk\",\"columns\":[\"id\"],"
                         + "\"enforced\":false,\"foreign_table\":\"parent\","
                         + "\"foreign_columns\":[\"a\",\"b\"]}]")) {
      TTableDetails details = details(metadata, false);
      details.setForeign_key_column_ordinals(
              Collections.singletonMap("fk", Collections.singletonList(0)));
      HeavyDBTable table = table(details);
      assertTrue(metadata, table.getStatistic().getKeys().isEmpty());
      assertTrue(metadata, table.getStatistic().getReferentialConstraints().isEmpty());
    }
  }

  @Test
  public void rowCountIsAvailableWithoutTrust() {
    TTableDetails details = details("[]", false);
    details.setNum_rows(0);
    HeavyDBTable.setTrustUnenforcedTableConstraints(false);
    assertEquals(Double.valueOf(0), table(details).getStatistic().getRowCount());
    details.setNum_rows(42);
    details.setKey_metainfo(unique());
    HeavyDBTable table = table(details);
    assertEquals(Double.valueOf(42), table.getStatistic().getRowCount());
    assertTrue(table.getStatistic().getKeys().isEmpty());
    details.unsetNum_rows();
    assertNull(table(details).getStatistic().getRowCount());
    details.setNum_rows(-1);
    assertNull(table(details).getStatistic().getRowCount());
  }

  @Test
  public void invalidationRemovesPublishedRowCounts() {
    TTableDetails details = details("[]", false);
    details.setNum_rows(42);
    table(details);
    assertEquals(Double.valueOf(42),
            HeavyDBTable.getRowCountEstimate(Arrays.asList("constraint_test", "child")));
    HeavyDBTable.invalidateMetadata("CONSTRAINT_TEST", "CHILD");
    assertNull(
            HeavyDBTable.getRowCountEstimate(Arrays.asList("constraint_test", "child")));
  }

  @Test
  public void unknownRowCountReplacesOldEstimate() {
    table(details("[]", false));
    TTableDetails unknown = details("[]", false);
    unknown.unsetNum_rows();
    table(unknown);
    assertNull(
            HeavyDBTable.getRowCountEstimate(Arrays.asList("constraint_test", "child")));
  }

  @Test
  @SuppressWarnings("unchecked")
  public void parentSchemaChangeInvalidatesReferringTables() throws Exception {
    Field field = MetaConnect.class.getDeclaredField("DB_TABLE_DETAILS");
    field.setAccessible(true);
    Map<List<String>, Table> cache = (Map<List<String>, Table>) field.get(null);
    List<String> childKey = Arrays.asList("CONSTRAINT_TEST", "CHILD");
    List<String> otherKey = Arrays.asList("CONSTRAINT_TEST", "OTHER");
    Table oldChild = cache.put(childKey, table(details(foreignKey(), false)));
    Table oldOther = cache.put(
            otherKey, new HeavyDBTable(details("[]", false), "constraint_test", "other"));
    try {
      new MetaConnect(0, "", null, null, null)
              .updateMetaData("constraint_test", "parent");
      assertFalse(cache.containsKey(childKey));
      assertTrue(cache.containsKey(otherKey));
      assertNull(HeavyDBTable.getRowCountEstimate(childKey));
    } finally {
      cache.remove(childKey);
      cache.remove(otherKey);
      if (oldChild != null) {
        cache.put(childKey, oldChild);
      }
      if (oldOther != null) {
        cache.put(otherKey, oldOther);
      }
    }
  }
}
