/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.mapd.calcite.parser;

import static org.junit.Assert.*;

import com.mapd.metadata.MetaConnect;

import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;
import org.apache.calcite.util.mapping.IntPair;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import ai.heavy.thrift.server.TColumnType;
import ai.heavy.thrift.server.TDatumType;
import ai.heavy.thrift.server.TTableDetails;
import ai.heavy.thrift.server.TTypeInfo;

public class MetaConnectTest {
  private static final String DATABASE = "metadata_publication_test";
  private static final String OTHER_DATABASE = "metadata_publication_other";
  private boolean previousEnabled;
  private boolean previousTrust;

  @Rule
  public TemporaryFolder temporary = new TemporaryFolder();

  @Before
  public void setup() {
    previousEnabled = HeavyDBTable.setOptimizerMetadataEnabled(true);
    previousTrust = HeavyDBTable.setTrustUnenforcedTableConstraints(true);
    invalidate(DATABASE, "");
    invalidate(OTHER_DATABASE, "");
  }

  @After
  public void teardown() {
    invalidate(DATABASE, "");
    invalidate(OTHER_DATABASE, "");
    HeavyDBTable.setOptimizerMetadataEnabled(previousEnabled);
    HeavyDBTable.setTrustUnenforcedTableConstraints(previousTrust);
  }

  private static void invalidate(String database, String table) {
    // Invalidation comes through another connection in the server as well.
    new MetaConnect(-1, "", null, null, null, database).updateMetaData(database, table);
  }

  private static TTableDetails details(long rows) {
    TColumnType column = new TColumnType();
    column.setCol_name("id");
    TTypeInfo type = new TTypeInfo();
    type.setType(TDatumType.INT);
    type.setNullable(false);
    column.setCol_type(type);
    TTableDetails result = new TTableDetails();
    result.setRow_desc(Collections.singletonList(column));
    result.setKey_metainfo("[]");
    result.setNum_rows(rows);
    return result;
  }

  private MetaConnect connection(Function<String, TTableDetails> fetch) {
    return new MetaConnect(
            -1, temporary.getRoot().getPath(), null, null, null, DATABASE) {
      @Override
      public TTableDetails get_table_details(String name) {
        return fetch.apply(name);
      }
    };
  }

  private void assertCachedRows(MetaConnect connection, Table table, double rows) {
    assertEquals(Double.valueOf(rows), table.getStatistic().getRowCount());
    assertSame(table, connection.getTable("child"));
    assertEquals(Double.valueOf(rows),
            HeavyDBTable.getRowCountEstimate(Arrays.asList(DATABASE, "child")));
  }

  private void invalidationDuringFetch(String invalidatedTable) {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      if (fetches.incrementAndGet() == 1) {
        TTableDetails stale = details(5);
        invalidate(DATABASE.toUpperCase(), invalidatedTable);
        return stale;
      }
      return details(8);
    });
    assertCachedRows(connection, connection.getTable("child"), 8);
    assertEquals(2, fetches.get());
  }

  @Test
  public void tableInvalidationRejectsInFlightDetails() {
    invalidationDuringFetch("CHILD");
  }

  @Test
  public void databaseInvalidationRejectsInFlightDetails() {
    invalidationDuringFetch("");
  }

  @Test
  public void parentInvalidationRejectsInFlightForeignKeyOrdinals() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      boolean first = fetches.incrementAndGet() == 1;
      TTableDetails result = details(8);
      result.setKey_metainfo("[{\"type\":\"FOREIGN KEY\",\"name\":\"fk\","
              + "\"columns\":[\"id\"],\"foreign_table\":\"parent\","
              + "\"foreign_columns\":[\"id\"],\"enforced\":false}]");
      result.setForeign_key_column_ordinals(
              Collections.singletonMap("fk", Collections.singletonList(first ? 0 : 2)));
      if (first) {
        invalidate(DATABASE, "parent");
      }
      return result;
    });
    Table table = connection.getTable("child");
    assertEquals(Collections.singletonList(IntPair.of(0, 2)),
            table.getStatistic().getReferentialConstraints().get(0).getColumnPairs());
    assertCachedRows(connection, table, 8);
    assertEquals(2, fetches.get());
  }

  @Test
  public void invalidationDuringViewFetchRejectsOldDetails() throws Exception {
    File catalog = new File(temporary.newFolder("catalogs"), DATABASE);
    try (Connection db = DriverManager.getConnection("jdbc:sqlite:" + catalog);
            Statement sql = db.createStatement()) {
      sql.execute("CREATE TABLE mapd_tables (tableid INT, name TEXT)");
      sql.execute("CREATE TABLE mapd_views (tableid INT, sql TEXT)");
      sql.execute("INSERT INTO mapd_tables VALUES (1, 'child')");
      sql.execute("INSERT INTO mapd_views VALUES (1, 'SELECT 1;')");
    }
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      boolean first = fetches.incrementAndGet() == 1;
      TTableDetails result = details(first ? 5 : 8);
      result.setView_sql("SELECT 1;");
      if (first) {
        invalidate(DATABASE, "parent");
      }
      return result;
    });
    Table table = connection.getTable("child");
    assertEquals(Schema.TableType.VIEW, table.getJdbcTableType());
    assertEquals("SELECT 1", ((HeavyDBView) table).getViewSql());
    assertCachedRows(connection, table, 8);
    assertEquals(2, fetches.get());
  }

  @Test
  public void otherDatabaseInvalidationDoesNotRetryFetch() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      fetches.incrementAndGet();
      invalidate(OTHER_DATABASE, "child");
      return details(8);
    });
    assertCachedRows(connection, connection.getTable("child"), 8);
    assertEquals(1, fetches.get());
  }

  @Test
  public void repeatedInvalidationsDoNotPublishIntermediateCounts() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      int attempt = fetches.incrementAndGet();
      if (attempt < 4) {
        invalidate(DATABASE, "child");
      }
      return details(attempt * 2);
    });
    assertCachedRows(connection, connection.getTable("child"), 8);
    assertEquals(4, fetches.get());
  }

  @Test
  public void failedFetchLeavesNoPublishedCount() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      if (fetches.incrementAndGet() == 1) {
        throw new IllegalStateException("metadata fetch failed");
      }
      return details(8);
    });
    try {
      connection.getTable("child");
      fail("Expected the fetch failure");
    } catch (IllegalStateException expected) {
      assertEquals("metadata fetch failed", expected.getMessage());
    }
    assertNull(HeavyDBTable.getRowCountEstimate(Arrays.asList(DATABASE, "child")));
    assertCachedRows(connection, connection.getTable("child"), 8);
    assertEquals(2, fetches.get());
  }

  private static void await(CountDownLatch latch) {
    try {
      assertTrue(
              "Timed out waiting for metadata fetch", latch.await(10, TimeUnit.SECONDS));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new AssertionError(e);
    }
  }

  private void competingFetches(boolean invalidateDuringFetch) throws Exception {
    CountDownLatch fetched = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    ExecutorService workers = Executors.newFixedThreadPool(2);
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect slow = connection(name -> {
      fetches.incrementAndGet();
      TTableDetails result = details(5);
      fetched.countDown();
      await(release);
      return result;
    });
    MetaConnect fast = connection(name -> {
      fetches.incrementAndGet();
      return details(8);
    });
    try {
      Future<Table> pending = workers.submit(() -> slow.getTable("child"));
      await(fetched);
      if (invalidateDuringFetch) {
        // Invalidation must complete while the metadata fetch remains blocked.
        workers.submit(() -> invalidate(DATABASE, "child")).get(10, TimeUnit.SECONDS);
      }
      Table winner =
              workers.submit(() -> fast.getTable("child")).get(10, TimeUnit.SECONDS);
      release.countDown();
      Table late = pending.get(10, TimeUnit.SECONDS);
      assertCachedRows(fast, winner, 8);
      assertSame(winner, late);
      assertEquals(2, fetches.get());
    } finally {
      release.countDown();
      workers.shutdownNow();
      assertTrue(workers.awaitTermination(10, TimeUnit.SECONDS));
    }
  }

  @Test
  public void invalidatedFetchCannotOverwriteNewlyPublishedCount() throws Exception {
    competingFetches(true);
  }

  @Test
  public void concurrentMissesReturnThePublishedTableAndCount() throws Exception {
    competingFetches(false);
  }
}
