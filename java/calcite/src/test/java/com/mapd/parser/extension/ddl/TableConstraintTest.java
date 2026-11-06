/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.mapd.parser.extension.ddl;

import static org.junit.Assert.*;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import org.junit.Test;

import ai.heavy.thrift.calciteserver.InvalidParseRequest;

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

  private static JsonArray strings(String... values) {
    JsonArray array = new JsonArray();
    for (String value : values) {
      array.add(value);
    }
    return array;
  }

  @Test
  public void createKeysRetainNamesColumnsAndReferences() throws Exception {
    JsonArray elements = payload("CREATE TABLE child (id INT, code INT, parent_id INT, "
            + "CONSTRAINT child_pk PRIMARY KEY(id), UNIQUE(code), "
            + "CONSTRAINT child_fk FOREIGN KEY(parent_id) REFERENCES parent(id));")
                                 .getAsJsonArray("elements");
    assertEquals(6, elements.size());
    JsonObject primary = elements.get(3).getAsJsonObject();
    assertEquals("child_pk", primary.get("name").getAsString());
    assertEquals("PRIMARY KEY", primary.get("constraintType").getAsString());
    assertEquals(strings("id"), primary.get("columns"));
    JsonObject unique = elements.get(4).getAsJsonObject();
    assertTrue(unique.get("name").isJsonNull());
    assertEquals("UNIQUE", unique.get("constraintType").getAsString());
    assertEquals(strings("code"), unique.get("columns"));
    JsonObject foreign = elements.get(5).getAsJsonObject();
    assertEquals("child_fk", foreign.get("name").getAsString());
    assertEquals("FOREIGN KEY", foreign.get("constraintType").getAsString());
    assertEquals(strings("parent_id"), foreign.get("columns"));
    JsonObject reference = foreign.getAsJsonObject("references");
    assertEquals("parent", reference.get("table").getAsString());
    assertEquals(strings("id"), reference.get("columns"));
  }

  @Test
  public void storageDirectivesCoexistWithMarkerNamedConstraints() throws Exception {
    JsonArray elements = payload("CREATE TABLE t (id INT, a TEXT, b TEXT, SHARD KEY(id), "
            + "SHARED DICTIONARY(b) REFERENCES t(a), "
            + "CONSTRAINT SHARD_KEY PRIMARY KEY(id), CONSTRAINT SHARED_DICT UNIQUE(id)) "
            + "WITH (SHARD_COUNT=2);")
                                 .getAsJsonArray("elements");
    assertEquals(7, elements.size());
    assertEquals("SHARD KEY",
            elements.get(3).getAsJsonObject().get("constraintType").getAsString());
    assertEquals("SHARED DICTIONARY",
            elements.get(4).getAsJsonObject().get("constraintType").getAsString());
    assertEquals("PRIMARY KEY",
            elements.get(5).getAsJsonObject().get("constraintType").getAsString());
    assertEquals(
            "SHARD_KEY", elements.get(5).getAsJsonObject().get("name").getAsString());
    assertEquals("UNIQUE",
            elements.get(6).getAsJsonObject().get("constraintType").getAsString());
    assertEquals(
            "SHARED_DICT", elements.get(6).getAsJsonObject().get("name").getAsString());
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

  private void assertReferenceColumnsRequired(String sql) {
    InvalidParseRequest error =
            assertThrows(sql, InvalidParseRequest.class, () -> processDdlCommand(sql));
    assertEquals(-2, error.getWhatUp());
    assertTrue(error.getWhyUp(), error.getWhyUp().contains("\"(\""));
  }

  @Test
  public void createForeignKeyRequiresReferenceColumns() {
    assertReferenceColumnsRequired(
            "CREATE TABLE child (id INT, FOREIGN KEY(id) REFERENCES parent);");
  }

  @Test
  public void createNamedForeignKeyRequiresReferenceColumns() {
    assertReferenceColumnsRequired("CREATE TABLE child (id INT, "
            + "CONSTRAINT fk FOREIGN KEY(id) REFERENCES parent);");
  }

  @Test
  public void addForeignKeyRequiresReferenceColumns() {
    assertReferenceColumnsRequired(
            "ALTER TABLE child ADD FOREIGN KEY(id) REFERENCES parent;");
  }

  @Test
  public void addNamedForeignKeyRequiresReferenceColumns() {
    assertReferenceColumnsRequired(
            "ALTER TABLE child ADD CONSTRAINT fk FOREIGN KEY(id) REFERENCES parent;");
  }

  @Test(expected = InvalidParseRequest.class)
  public void createForeignKeyRejectsEmptyReferenceColumns() throws Exception {
    processDdlCommand(
            "CREATE TABLE child (id INT, FOREIGN KEY(id) REFERENCES parent());");
  }

  @Test(expected = InvalidParseRequest.class)
  public void addForeignKeyRejectsEmptyReferenceColumns() throws Exception {
    processDdlCommand("ALTER TABLE child ADD FOREIGN KEY(id) REFERENCES parent();");
  }

  @Test
  public void compositeForeignKeysPreserveReferenceColumnOrder() throws Exception {
    JsonObject create = payload("CREATE TABLE child (a INT, b INT, "
            + "CONSTRAINT fk FOREIGN KEY(b, a) REFERENCES parent(y, x));")
                                .getAsJsonArray("elements")
                                .get(2)
                                .getAsJsonObject();
    JsonObject alter = payload("ALTER TABLE child ADD CONSTRAINT fk "
            + "FOREIGN KEY(b, a) REFERENCES parent(y, x);")
                               .getAsJsonArray("constraintData")
                               .get(0)
                               .getAsJsonObject();
    assertEquals(create, alter);
    assertEquals(strings("b", "a"), create.get("columns"));
    assertEquals(strings("y", "x"), create.getAsJsonObject("references").get("columns"));
  }

  @Test
  public void sharedDictionaryMayReferenceLocalColumn() throws Exception {
    JsonObject key =
            payload("CREATE TABLE t (a TEXT, b TEXT, SHARED DICTIONARY(b) REFERENCES a);")
                    .getAsJsonArray("elements")
                    .get(2)
                    .getAsJsonObject();
    assertEquals("SHARED DICTIONARY", key.get("constraintType").getAsString());
    JsonObject reference = key.getAsJsonObject("references");
    assertFalse(reference.has("table"));
    assertEquals(strings("a"), reference.get("columns"));
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
  public void addForeignKeyRetainsNameColumnsAndReference() throws Exception {
    JsonObject add = payload("ALTER TABLE child ADD CONSTRAINT child_fk "
            + "FOREIGN KEY(parent_id) REFERENCES parent(id);");
    assertEquals("ADD_CONSTRAINT", add.get("alterType").getAsString());
    JsonArray constraints = add.getAsJsonArray("constraintData");
    assertEquals(1, constraints.size());
    JsonObject key = constraints.get(0).getAsJsonObject();
    assertEquals("child_fk", key.get("name").getAsString());
    assertEquals("FOREIGN KEY", key.get("constraintType").getAsString());
    assertEquals(strings("parent_id"), key.get("columns"));
    JsonObject reference = key.getAsJsonObject("references");
    assertEquals("parent", reference.get("table").getAsString());
    assertEquals(strings("id"), reference.get("columns"));
  }

  @Test
  public void addUnnamedCompositePrimaryKeyPreservesColumnOrder() throws Exception {
    JsonObject add = payload("ALTER TABLE t ADD PRIMARY KEY(a, b);");
    assertEquals("ADD_CONSTRAINT", add.get("alterType").getAsString());
    JsonArray constraints = add.getAsJsonArray("constraintData");
    assertEquals(1, constraints.size());
    JsonObject key = constraints.get(0).getAsJsonObject();
    assertTrue(key.get("name").isJsonNull());
    assertEquals("PRIMARY KEY", key.get("constraintType").getAsString());
    assertEquals(strings("a", "b"), key.get("columns"));
  }

  @Test
  public void addColumnIsNotAConstraint() throws Exception {
    assertEquals("ADD_COLUMN",
            payload("ALTER TABLE t ADD COLUMN c INT;").get("alterType").getAsString());
  }

  @Test
  public void dropColumnIsNotAConstraint() throws Exception {
    assertEquals("DROP_COLUMN",
            payload("ALTER TABLE t DROP COLUMN c;").get("alterType").getAsString());
  }

  @Test
  public void ordinaryAlterIsUnchanged() throws Exception {
    assertEquals("RENAME_COLUMN",
            payload("ALTER TABLE t RENAME COLUMN id TO other;")
                    .get("alterType")
                    .getAsString());
  }
}
