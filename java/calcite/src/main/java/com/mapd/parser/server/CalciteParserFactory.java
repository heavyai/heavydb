/*
 * Copyright 2017 MapD Technologies, Inc.
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
package com.mapd.parser.server;

import com.mapd.calcite.parser.MapDParser;
import com.mapd.calcite.parser.MapDSqlOperatorTable;
import com.mapd.common.SockTransportProperties;

import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.util.ConversionUtil;
import org.apache.commons.pool.PoolableObjectFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.function.Supplier;

/**
 *
 * @author michael
 */
class CalciteParserFactory implements PoolableObjectFactory {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteParserFactory.class);

  private final String dataDir;
  private final Map<String, ExtensionFunction> extSigs;
  private final int mapdPort;
  private final SockTransportProperties socket_transport_properties;
  private volatile MapDSqlOperatorTable tableOperator;
  private final Supplier<MapDSqlOperatorTable> tableOperatorSupplier =
          new Supplier<MapDSqlOperatorTable>() {
            @Override
            public MapDSqlOperatorTable get() {
              return tableOperator;
            }
          };

  static {
    System.setProperty(
            "saffron.default.charset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
    System.setProperty(
            "saffron.default.nationalcharset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
    System.setProperty("saffron.default.collation.name",
            ConversionUtil.NATIVE_UTF16_CHARSET_NAME + "$en_US");
  }

  public CalciteParserFactory(String dataDir,
          final Map<String, ExtensionFunction> extSigs,
          int mapdPort,
          SockTransportProperties skT) {
    this.dataDir = dataDir;
    this.extSigs = extSigs;
    this.mapdPort = mapdPort;
    this.socket_transport_properties = skT;

    updateOperatorTable();
  }

  public void updateOperatorTable() {
    MapDSqlOperatorTable tableOperator =
            new MapDSqlOperatorTable(SqlStdOperatorTable.instance());
    MapDSqlOperatorTable.addUDF(tableOperator, extSigs);
    this.tableOperator = tableOperator;
  }

  @Override
  public Object makeObject() throws Exception {
    MapDParser obj = new MapDParser(
            dataDir, tableOperatorSupplier, mapdPort, socket_transport_properties);
    return obj;
  }

  @Override
  public void destroyObject(Object obj) throws Exception {
    // no need to do anything
  }

  @Override
  public boolean validateObject(Object obj) {
    MapDParser mdp = (MapDParser) obj;
    if (mdp.getCallCount() < 1000) {
      return true;
    } else {
      MAPDLOGGER.debug(" invalidating object due to max use count");
      return false;
    }
  }

  @Override
  public void activateObject(Object obj) throws Exception {
    // don't need to do anything
  }

  @Override
  public void passivateObject(Object obj) throws Exception {
    // nothing to currently do here
  }
}
