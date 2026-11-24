/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.mapd.calcite.parser;

import static org.junit.Assert.*;

import com.mapd.common.SockTransportProperties;
import com.mapd.metadata.MetaConnect;
import com.mapd.parser.server.CalciteServerHandler;

import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.util.mapping.IntPair;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TField;
import org.apache.thrift.protocol.TMessage;
import org.apache.thrift.protocol.TMessageType;
import org.apache.thrift.protocol.TStruct;
import org.apache.thrift.protocol.TType;
import org.apache.thrift.transport.TMemoryBuffer;
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
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

import ai.heavy.thrift.calciteserver.CalciteServer;
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

  private static void invalidateStatistics(String database, String table) {
    new MetaConnect(-1, "", null, null, null, database)
            .updateMetaData(database, table, true);
  }

  private void invalidateOverThrift(String table, Boolean statisticsOnly) throws Exception {
    CalciteServerHandler handler = new CalciteServerHandler(-1,
            temporary.getRoot().getPath(),
            "src/test/resources/ast/test_extension_functions.ast",
            SockTransportProperties.getUnencryptedClient(),
            "");
    TMemoryBuffer request = new TMemoryBuffer(256);
    TMemoryBuffer response = new TMemoryBuffer(256);
    CalciteServer.Client client = new CalciteServer.Client(
            new TBinaryProtocol(response), new TBinaryProtocol(request));
    if (statisticsOnly != null) {
      client.send_updateMetadata(DATABASE, table, statisticsOnly);
    } else {
      // A client built against the previous IDL sends no statistics field.
      TBinaryProtocol protocol = new TBinaryProtocol(request);
      protocol.writeMessageBegin(new TMessage("updateMetadata", TMessageType.CALL, 0));
      protocol.writeStructBegin(new TStruct("updateMetadata_args"));
      protocol.writeFieldBegin(new TField("catalog", TType.STRING, (short) 1));
      protocol.writeString(DATABASE);
      protocol.writeFieldEnd();
      protocol.writeFieldBegin(new TField("table", TType.STRING, (short) 2));
      protocol.writeString(table);
      protocol.writeFieldEnd();
      protocol.writeFieldStop();
      protocol.writeStructEnd();
      protocol.writeMessageEnd();
    }
    new CalciteServer.Processor<>(handler).process(
            new TBinaryProtocol(request), new TBinaryProtocol(response));
    client.recv_updateMetadata();
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
  public void statisticsInvalidationPreservesUnchangedMetadata() throws Exception {
    checkMetadataScopes(false, false);
  }

  @Test
  public void thriftRoutesStatisticsAndSchemaInvalidation() throws Exception {
    checkMetadataScopes(true, false);
  }

  @Test
  public void thriftWithoutStatisticsFieldStillInvalidatesSchema() throws Exception {
    checkMetadataScopes(true, null);
  }

  private void checkMetadataScopes(boolean useThrift, Boolean schemaScope) throws Exception {
    File catalog = new File(temporary.newFolder("catalogs"), DATABASE);
    try (Connection db = DriverManager.getConnection("jdbc:sqlite:" + catalog);
            Statement sql = db.createStatement()) {
      sql.execute("CREATE TABLE mapd_tables (tableid INT, name TEXT)");
      sql.execute("CREATE TABLE mapd_views (tableid INT, sql TEXT)");
      sql.execute("INSERT INTO mapd_tables VALUES (1, 'parent'), (2, 'child'),"
              + " (3, 'view'), (4, 'unrelated')");
      sql.execute("INSERT INTO mapd_views VALUES (3, 'SELECT 1;')");
    }
    AtomicInteger rows = new AtomicInteger(5);
    MetaConnect connection = connection(name -> {
      TTableDetails result = details(rows.get());
      if (name.equals("view")) result.setView_sql("SELECT 1;");
      if (name.equals("child")) {
        result.setKey_metainfo("[{\"type\":\"FOREIGN KEY\",\"name\":\"fk\","
                + "\"columns\":[\"id\"],\"foreign_table\":\"parent\","
                + "\"foreign_columns\":[\"id\"],\"enforced\":false}]");
        result.setForeign_key_column_ordinals(
                Collections.singletonMap("fk", Collections.singletonList(0)));
      }
      return result;
    });
    Table parent = connection.getTable("parent");
    Table child = connection.getTable("child");
    Table view = connection.getTable("view");
    Table unrelated = connection.getTable("unrelated");
    Set<String> tables = connection.getTables();
    assertEquals(4, tables.size());
    assertEquals(1, child.getStatistic().getReferentialConstraints().size());

    rows.set(8);
    if (useThrift) {
      invalidateOverThrift("PARENT", true);
    } else {
      invalidateStatistics(DATABASE.toUpperCase(), "PARENT");
    }
    assertNull(HeavyDBTable.getRowCountEstimate(Arrays.asList(DATABASE, "parent")));
    assertSame(tables, connection.getTables());
    assertSame(child, connection.getTable("child"));
    assertSame(view, connection.getTable("view"));
    assertSame(unrelated, connection.getTable("unrelated"));
    Table refreshed = connection.getTable("parent");
    assertNotSame(parent, refreshed);
    assertEquals(Double.valueOf(8), refreshed.getStatistic().getRowCount());
    assertEquals(Double.valueOf(8),
            HeavyDBTable.getRowCountEstimate(Arrays.asList(DATABASE, "parent")));
    assertEquals(Double.valueOf(5), parent.getStatistic().getRowCount());
    assertEquals(Double.valueOf(5),
            HeavyDBTable.getRowCountEstimate(Arrays.asList(DATABASE, "child")));

    // DDL still retires dependent metadata, even after a statistics-only refresh.
    if (useThrift) {
      invalidateOverThrift("parent", schemaScope);
    } else {
      invalidate(DATABASE, "parent");
    }
    assertNotSame(refreshed, connection.getTable("parent"));
    assertNotSame(child, connection.getTable("child"));
    assertNotSame(view, connection.getTable("view"));
    assertNotSame(tables, connection.getTables());
    assertSame(unrelated, connection.getTable("unrelated"));
  }

  @Test
  public void expandedViewsFollowTheirBaseTableStatistics() throws Exception {
    File catalog = new File(temporary.newFolder("catalogs"), DATABASE);
    try (Connection db = DriverManager.getConnection("jdbc:sqlite:" + catalog);
            Statement sql = db.createStatement()) {
      sql.execute("CREATE TABLE mapd_tables (tableid INT, name TEXT)");
      sql.execute("CREATE TABLE mapd_views (tableid INT, sql TEXT)");
      sql.execute(
              "INSERT INTO mapd_tables VALUES (1,'base_t'), (2,'inner_v'), (3,'outer_v')");
      sql.execute("INSERT INTO mapd_views VALUES (2,'SELECT id FROM base_t;'),"
              + "(3,'SELECT id FROM inner_v;')");
    }
    AtomicInteger rows = new AtomicInteger(0);
    AtomicReference<Runnable> whileBindingScan = new AtomicReference<>();
    MetaConnect connection = connection(name -> {
      TTableDetails td = details(rows.get());
      if (name.equals("base_t")) {
        TColumnType column = td.row_desc.get(0);
        column.setCol_type(new TTypeInfo(column.col_type) {
          @Override
          public TDatumType getType() {
            Runnable hook = whileBindingScan.getAndSet(null);
            if (hook != null) hook.run();
            return super.getType();
          }
        });
      } else {
        td.setView_sql("view");
      }
      return td;
    });
    connection.getTables();
    connection.getTable("base_t");
    HeavyDBView inner = (HeavyDBView) connection.getTable("inner_v");
    HeavyDBView outer = (HeavyDBView) connection.getTable("outer_v");
    HeavyDBParser parser = new HeavyDBParser(temporary.getRoot().getPath(),
            ()
                    -> new HeavyDBSqlOperatorTable(SqlStdOperatorTable.instance()),
            -1,
            null);
    parser.setUser(new HeavyDBUser("admin", "", DATABASE, -1, Collections.emptyList()));
    HeavyDBParser.CURRENT_PARSER.set(parser);
    try {
      RelNode original = outer.toRel(null, null);
      assertEquals(Double.valueOf(0), scanRows(original));
      invalidateStatistics(OTHER_DATABASE, "base_t");
      invalidateStatistics(DATABASE, "unrelated");
      assertSame(original, outer.toRel(null, null));

      rows.set(6);
      invalidateStatistics(DATABASE, "BASE_T");
      assertEquals(Double.valueOf(6),
              connection.getTable("base_t").getStatistic().getRowCount());
      RelNode refreshed = outer.toRel(null, null);
      assertEquals(Double.valueOf(6), scanRows(refreshed));
      assertEquals(Double.valueOf(6), scanRows(inner.toRel(null, null)));
      assertSame(refreshed, outer.toRel(null, null));
      assertEquals(Double.valueOf(0),
              scanRows(original)); // Existing planners keep their snapshot.

      // Replace the source while Calcite binds it, before the view plan is published.
      rows.set(7);
      invalidateStatistics(DATABASE, "base_t");
      connection.getTable("base_t");
      whileBindingScan.set(() -> {
        rows.set(8);
        invalidateStatistics(DATABASE, "base_t");
        connection.getTable("base_t");
      });
      outer.toRel(null, null);
      assertNull(whileBindingScan.get());
      assertEquals(Double.valueOf(8), scanRows(outer.toRel(null, null)));
    } finally {
      HeavyDBParser.CURRENT_PARSER.remove();
    }
  }

  private static Double scanRows(RelNode root) {
    Set<RelOptTable> tables = RelOptUtil.findTables(root);
    assertEquals(1, tables.size());
    return tables.iterator()
            .next()
            .unwrap(HeavyDBTable.class)
            .getStatistic()
            .getRowCount();
  }

  @Test
  public void statisticsInvalidationRequiresTableName() {
    MetaConnect connection = connection(name -> details(5));
    Table child = connection.getTable("child");
    assertThrows(IllegalArgumentException.class, () -> invalidateStatistics(DATABASE, ""));
    assertCachedRows(connection, child, 5);
  }

  @Test
  public void statisticsInvalidationRejectsInFlightCount() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      if (fetches.incrementAndGet() == 1) {
        invalidateStatistics(DATABASE, "child");
        return details(5);
      }
      return details(8);
    });
    assertCachedRows(connection, connection.getTable("child"), 8);
    assertEquals(2, fetches.get());
  }

  @Test
  public void statisticsInvalidationDoesNotRestartUnrelatedFetch() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      if (fetches.incrementAndGet() == 1) {
        invalidateStatistics(DATABASE, "parent");
      }
      return details(8);
    });
    assertCachedRows(connection, connection.getTable("child"), 8);
    assertEquals(1, fetches.get());
  }

  @Test
  public void statisticsInvalidationInOtherDatabaseDoesNotRestartFetch() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      fetches.incrementAndGet();
      invalidateStatistics(OTHER_DATABASE, "child");
      return details(8);
    });
    assertCachedRows(connection, connection.getTable("child"), 8);
    assertEquals(1, fetches.get());
  }

  @Test
  public void schemaInvalidationStillRejectsFetchAfterStatisticsInvalidation() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      if (fetches.incrementAndGet() == 1) {
        invalidateStatistics(DATABASE, "child");
        invalidate(DATABASE, "parent");
        return details(5);
      }
      return details(8);
    });
    assertCachedRows(connection, connection.getTable("child"), 8);
    assertEquals(2, fetches.get());
  }

  @Test
  public void failedStatisticsRefetchDoesNotRepublishOldCount() {
    AtomicInteger fetches = new AtomicInteger();
    MetaConnect connection = connection(name -> {
      int attempt = fetches.incrementAndGet();
      if (attempt == 2) throw new IllegalStateException("metadata fetch failed");
      return details(attempt == 1 ? 5 : 8);
    });
    Table old = connection.getTable("child");
    invalidateStatistics(DATABASE, "child");
    try {
      connection.getTable("child");
      fail("Expected the fetch failure");
    } catch (IllegalStateException expected) {
      assertEquals("metadata fetch failed", expected.getMessage());
    }
    assertNull(HeavyDBTable.getRowCountEstimate(Arrays.asList(DATABASE, "child")));
    assertEquals(Double.valueOf(5), old.getStatistic().getRowCount());
    assertCachedRows(connection, connection.getTable("child"), 8);
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

  private void competingFetches(boolean invalidateDuringFetch, boolean statisticsOnly)
          throws Exception {
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
        workers.submit(() -> {
                 if (statisticsOnly) {
                   invalidateStatistics(DATABASE, "child");
                 } else {
                   invalidate(DATABASE, "child");
                 }
               }).get(10, TimeUnit.SECONDS);
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
    competingFetches(true, false);
  }

  @Test
  public void concurrentMissesReturnThePublishedTableAndCount() throws Exception {
    competingFetches(false, false);
  }

  @Test
  public void statisticsInvalidatedFetchCannotOverwriteNewCount() throws Exception {
    competingFetches(true, true);
  }
}
