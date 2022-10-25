/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mapd.calcite.parser;

import com.mapd.common.SockTransportProperties;
import com.mapd.metadata.MetaConnect;

import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.SchemaVersion;
import org.apache.calcite.schema.Table;
import org.apache.calcite.util.ConversionUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

public class HeavyDBSchema implements Schema {
  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(HeavyDBSchema.class);

  final private MetaConnect metaConnect;
  private SockTransportProperties sock_transport_properties = null;
  public HeavyDBSchema(String dataDir,
          HeavyDBParser dbParser,
          int dbPort,
          HeavyDBUser dbUser,
          SockTransportProperties skT,
          String db) {
    System.setProperty(
            "saffron.default.charset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
    System.setProperty(
            "saffron.default.nationalcharset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
    System.setProperty("saffron.default.collation.name",
            ConversionUtil.NATIVE_UTF16_CHARSET_NAME + "$en_US");
    metaConnect = new MetaConnect(dbPort, dataDir, dbUser, dbParser, skT, db);
  }

  @Override
  public Table getTable(String string) {
    Table table = metaConnect.getTable(string);
    return table;
  }

  @Override
  public Set<String> getTableNames() {
    Set<String> tableSet = metaConnect.getTables();
    return tableSet;
  }

  @Override
  public Collection<Function> getFunctions(String string) {
    Collection<Function> functionCollection = new HashSet<Function>();
    return functionCollection;
  }

  @Override
  public Set<String> getFunctionNames() {
    Set<String> functionSet = new HashSet<String>();
    return functionSet;
  }

  @Override
  public Schema getSubSchema(String string) {
    return null;
  }

  @Override
  public Set<String> getSubSchemaNames() {
    Set<String> hs = new HashSet<String>();
    return hs;
  }

  @Override
  public Expression getExpression(SchemaPlus sp, String string) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isMutable() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  void updateMetaData(String schema, String table) {
    metaConnect.updateMetaData(schema, table);
  }

  @Override
  public Schema snapshot(SchemaVersion sv) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelProtoDataType getType(String arg0) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public Set<String> getTypeNames() {
    throw new UnsupportedOperationException("Not supported yet.");
  }
}
