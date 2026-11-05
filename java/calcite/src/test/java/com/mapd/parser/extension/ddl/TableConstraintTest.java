/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.mapd.parser.extension.ddl;

import static org.junit.Assert.*;

import com.google.gson.JsonObject;

import org.junit.Test;

public class TableConstraintTest extends DDLTest {
  public TableConstraintTest() {
    resourceDirPath =
            TableConstraintTest.class.getClassLoader().getResource("").getPath();
  }

  private JsonObject payload(String sql) throws Exception {
    return gson.fromJson(processDdlCommand(sql).plan_result, JsonObject.class)
            .getAsJsonObject("payload");
  }

  private JsonObject constraint(String sql) throws Exception {
    return payload(sql).getAsJsonArray("elements").get(1).getAsJsonObject();
  }

  @Test
  public void primaryKeyIsNotShardDirective() throws Exception {
    JsonObject key =
            constraint("CREATE TABLE t (id INT, CONSTRAINT SHARD_KEY PRIMARY KEY(id));");
    assertEquals("PRIMARY KEY", key.get("constraintType").getAsString());
    assertEquals("SHARD_KEY", key.get("name").getAsString());
  }

  @Test
  public void uniqueIsNotSharedDictionaryDirective() throws Exception {
    JsonObject key =
            constraint("CREATE TABLE t (id INT, CONSTRAINT SHARED_DICT UNIQUE(id));");
    assertEquals("UNIQUE", key.get("constraintType").getAsString());
  }

  @Test
  public void shardDirectiveHasItsOwnType() throws Exception {
    assertEquals("SHARD KEY",
            constraint("CREATE TABLE t (id INT, SHARD KEY(id)) WITH (SHARD_COUNT=2);")
                    .get("constraintType")
                    .getAsString());
  }

  @Test
  public void sharedDictionaryHasItsOwnType() throws Exception {
    assertEquals("SHARED DICTIONARY",
            constraint("CREATE TABLE t (id TEXT, SHARED DICTIONARY(id) REFERENCES "
                    + "other(id));")
                    .get("constraintType")
                    .getAsString());
  }

  @Test
  public void foreignKeyRetainsReference() throws Exception {
    JsonObject key =
            constraint("CREATE TABLE t (id INT, FOREIGN KEY(id) REFERENCES parent(id));");
    assertEquals("FOREIGN KEY", key.get("constraintType").getAsString());
    assertEquals("parent", key.getAsJsonObject("references").get("table").getAsString());
  }

  @Test
  public void addAndDropHaveDistinctPayloads() throws Exception {
    JsonObject add = payload("ALTER TABLE t ADD CONSTRAINT uk UNIQUE(id);");
    assertEquals("ADD_CONSTRAINT", add.get("alterType").getAsString());
    assertEquals("UNIQUE",
            add.getAsJsonArray("constraintData")
                    .get(0)
                    .getAsJsonObject()
                    .get("constraintType")
                    .getAsString());
    JsonObject drop = payload("ALTER TABLE t DROP CONSTRAINT uk;");
    assertEquals("DROP_CONSTRAINT", drop.get("alterType").getAsString());
    assertEquals("uk", drop.get("constraintName").getAsString());
  }

  @Test
  public void ordinaryAlterIsUnchanged() throws Exception {
    assertEquals("RENAME_COLUMN",
            payload("ALTER TABLE t RENAME COLUMN id TO other;")
                    .get("alterType")
                    .getAsString());
  }
}
