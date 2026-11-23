/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.calcite.parser;

import com.mapd.common.SockTransportProperties;
import com.mapd.metadata.MetaConnect;

import org.apache.calcite.rel.RelReferentialConstraint;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.IntPair;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.transport.TTransport;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import ai.heavy.thrift.server.Heavy;
import ai.heavy.thrift.server.TExecuteMode;
import ai.heavy.thrift.server.TTableDetails;

/** Run by TableStatisticsTest against its real DBHandler over loopback Thrift. */
public class PlannerMetadataProbe {
  private final Heavy.Client client;
  private final String session;
  private final String database;
  private final boolean rewrites;
  private final boolean sharded;
  private final MetaConnect metadata;

  private PlannerMetadataProbe(Heavy.Client client,
          String session,
          String database,
          int port,
          boolean rewrites,
          boolean sharded) throws Exception {
    this.client = client;
    this.session = session;
    this.database = database;
    this.rewrites = rewrites;
    this.sharded = sharded;
    HeavyDBTable.setOptimizerMetadataEnabled(rewrites);
    HeavyDBTable.setTrustUnenforcedTableConstraints(true);
    metadata = new MetaConnect(port,
            "",
            new HeavyDBUser("admin", session, database, port, Collections.emptyList()),
            null,
            SockTransportProperties.getUnencryptedClient());
  }

  private static void equal(Object expected, Object actual, String what) {
    if (!Objects.equals(expected, actual)) {
      throw new AssertionError(what + ": expected " + expected + ", got " + actual);
    }
  }

  private void sql(String query) throws Exception {
    client.sql_execute(session, query, true, "", -1, -1);
  }

  private HeavyDBTable table(String name) {
    return (HeavyDBTable) metadata.getTable(name);
  }

  private void expectCount(long count) throws Exception {
    equal(count,
            client.sql_execute(session,
                          "SELECT COUNT(*) FROM planner_counts;",
                          true,
                          "",
                          -1,
                          -1)
                    .row_set.columns.get(0)
                    .data.int_col.get(0),
            "visible rows");
  }

  private void expectEstimate(long count) throws Exception {
    TTableDetails details = metadata.get_table_details("planner_counts");
    equal(rewrites, details.isSetNum_rows(), "native row estimate gate");
    if (rewrites) equal(count, details.getNum_rows(), "native physical rows");
    // This JVM is the metadata consumer, not the fixture's Calcite process.
    // Deliver the same invalidation it receives through updateMetadata RPCs.
    metadata.updateMetaData(database, "planner_counts");
    HeavyDBTable table = table("planner_counts");
    equal(rewrites ? Double.valueOf(count) : null,
            table.getStatistic().getRowCount(),
            "Java row estimate");
    equal(rewrites ? Double.valueOf(count) : null,
            HeavyDBTable.getRowCountEstimate(Arrays.asList(database, "planner_counts")),
            "qualified-name row estimate");
    equal(table, table("planner_counts"), "cached table");
    TTableDetails publicDetails = client.get_table_details(session, "planner_counts");
    equal(false, publicDetails.isSetNum_rows(), "public details omit row estimate");
    equal(false,
            publicDetails.isSetForeign_key_column_ordinals(),
            "public details omit planner ordinals");
  }

  private void rowCounts() throws Exception {
    try {
      sql("CREATE TABLE planner_counts (id INT" + (sharded ? ", SHARD KEY(id)" : "")
              + ") WITH (FRAGMENT_SIZE=2" + (sharded ? ", SHARD_COUNT=4" : "") + ");");
      expectCount(0);
      expectEstimate(0);
      sql("INSERT INTO planner_counts VALUES (1),(2),(3),(4),(5),(6);");
      expectCount(6);
      expectEstimate(6);
      sql("DELETE FROM planner_counts WHERE id <= 2;");
      expectCount(4);
      expectEstimate(6); // Storage estimates include unvacuumed deleted slots.
      sql("OPTIMIZE TABLE planner_counts WITH (VACUUM='true');");
      expectCount(4);
      expectEstimate(4);
      sql("TRUNCATE TABLE planner_counts;");
      expectCount(0);
      expectEstimate(0);
      HeavyDBTable.setTrustUnenforcedTableConstraints(false);
      sql("INSERT INTO planner_counts VALUES (7),(8);");
      expectCount(2);
      expectEstimate(2); // Row estimates do not require trusting declared keys.
    } finally {
      sql("DROP TABLE IF EXISTS planner_counts;");
    }
  }

  private void expectKeys(int parentA, int parentB) {
    TTableDetails details = metadata.get_table_details("planner_child");
    equal(rewrites,
            details.isSetForeign_key_column_ordinals(),
            "native foreign key metadata gate");
    if (rewrites) {
      equal(Arrays.asList(parentA, parentB),
              details.getForeign_key_column_ordinals().get("child_fk"),
              "native referenced-column ordinals");
    }
    Statistic parent = table("planner_parent").getStatistic();
    Statistic child = table("planner_child").getStatistic();
    equal(rewrites,
            parent.isKey(ImmutableBitSet.of(parentA, parentB)),
            "parent unique key");
    List<RelReferentialConstraint> keys = child.getReferentialConstraints();
    if (rewrites) {
      equal(1, keys.size(), "foreign key count");
      equal(Arrays.asList(database, "planner_child"),
              keys.get(0).getSourceQualifiedName(),
              "foreign key source");
      equal(Arrays.asList(database, "planner_parent"),
              keys.get(0).getTargetQualifiedName(),
              "foreign key target");
      equal(Arrays.asList(IntPair.of(1, parentA), IntPair.of(0, parentB)),
              keys.get(0).getColumnPairs(),
              "declared column mapping, not sorted column sets");
    } else {
      expectNoForeignKeys(child);
    }
  }

  private void expectNoForeignKeys(Statistic statistics) {
    List<RelReferentialConstraint> keys = statistics.getReferentialConstraints();
    // Calcite's UNKNOWN statistics use null, not an empty list.
    equal(true, keys == null || keys.isEmpty(), "no usable foreign key proof");
  }

  private void foreignKeys() throws Exception {
    try {
      sql("CREATE TABLE planner_parent (unused INT, shape POINT, a INT NOT NULL, "
              + "b INT NOT NULL, UNIQUE(a,b)" + (sharded ? ", SHARD KEY(a)" : "") + ")"
              + (sharded ? " WITH (SHARD_COUNT=4)" : "") + ";");
      sql("CREATE TABLE planner_child (x INT NOT NULL, y INT NOT NULL, "
              + "CONSTRAINT child_fk FOREIGN KEY(y,x) REFERENCES planner_parent(a,b)"
              + (sharded ? ", SHARD KEY(y)" : "") + ")"
              + (sharded ? " WITH (SHARD_COUNT=4)" : "") + ";");
      expectKeys(2, 3); // The POINT's physical columns are not planner columns.
      HeavyDBTable oldChild = table("planner_child");
      sql("ALTER TABLE planner_parent DROP COLUMN unused;");
      metadata.updateMetaData(database, "planner_parent");
      expectKeys(1, 2); // Recompute positions; do not retain persisted column IDs.
      equal(false, oldChild == table("planner_child"), "dependent child refreshed");
      HeavyDBTable.setTrustUnenforcedTableConstraints(false);
      equal(false,
              table("planner_parent").getStatistic().isKey(ImmutableBitSet.of(1, 2)),
              "untrusted unique key");
      expectNoForeignKeys(table("planner_child").getStatistic());
      HeavyDBTable.setTrustUnenforcedTableConstraints(true);
      HeavyDBTable.setOptimizerMetadataEnabled(false);
      equal(false,
              table("planner_parent").getStatistic().isKey(ImmutableBitSet.of(1, 2)),
              "disabled unique key");
      expectNoForeignKeys(table("planner_child").getStatistic());
    } finally {
      sql("DROP TABLE IF EXISTS planner_child;");
      sql("DROP TABLE IF EXISTS planner_parent;");
    }
  }

  public static void main(String[] args) throws Exception {
    int port = Integer.parseInt(args[0]);
    String database = args[1];
    try (TTransport transport =
                    SockTransportProperties.getUnencryptedClient().openClientTransport(
                            "localhost", port)) {
      if (!transport.isOpen()) transport.open();
      Heavy.Client client = new Heavy.Client(new TBinaryProtocol(transport));
      String session = client.connect("admin", "HyperInteractive", database);
      try {
        client.set_execution_mode(session, TExecuteMode.CPU);
        PlannerMetadataProbe probe = new PlannerMetadataProbe(client,
                session,
                database,
                port,
                Boolean.parseBoolean(args[3]),
                Boolean.parseBoolean(args[4]));
        if (args[2].equals("rows")) {
          probe.rowCounts();
        } else if (args[2].equals("keys")) {
          probe.foreignKeys();
        } else {
          throw new IllegalArgumentException("Unknown probe: " + args[2]);
        }
        System.out.println("Planner metadata probe passed: " + String.join(" ", args));
      } finally {
        client.disconnect(session);
      }
    }
  }
}
