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
package com.omnisci.jdbc;

import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TMapDException;
import com.mapd.thrift.server.TServerStatus;
import java.sql.Array;
import java.sql.Blob;
import java.sql.CallableStatement;
import java.sql.Clob;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.NClob;
import java.sql.PreparedStatement;
import java.sql.SQLClientInfoException;
import java.sql.SQLException;
import java.sql.SQLWarning;
import java.sql.SQLXML;
import java.sql.Savepoint;
import java.sql.Statement;
import java.sql.Struct;
import java.util.*;
import java.util.concurrent.Executor;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.mapd.common.SockTransportProperties;

/**
 *
 * @author michael
 */
/*
 * Param_pair - Simple pair class to hold the label for a component in the url
 * and an index into the url to that component.
 * For example in the url jdbc:mapd:hostname:6278 a Param_pair for
 * hostname would have a label of "hostname" and an index of 2

 */
class Param_pair {
  public Param_pair(String l, int i) {
    label = l;
    index = i;
  }
  public String label;
  public int index;
}
enum Connection_enums {
  host_name,
  port_num,
  db_name,
  protocol,
  key_store,
  key_store_pwd,
  user,
  user_passwd
}

public class OmniSciConnection implements java.sql.Connection {
  final static Logger logger = LoggerFactory.getLogger(OmniSciConnection.class);
  // A simple internal class to hold a summary of the properties passed to the connection
  // Properties can come two ways - via the url or via a Properties param
  class Connection_properties extends Hashtable<Connection_enums, Object> {
    // All 'used' properties should be listed in this enum map
    EnumMap<Connection_enums, Param_pair> connection_map =
            new EnumMap<Connection_enums, Param_pair>(Connection_enums.class) {
              {
                // the map allows peoperties to be access via a enum rather than string
                put(Connection_enums.host_name, new Param_pair("host_name", 2));
                put(Connection_enums.port_num, new Param_pair("port_num", 3));
                put(Connection_enums.db_name, new Param_pair("db_name", 4));
                put(Connection_enums.protocol, new Param_pair("protocol", 5));
                put(Connection_enums.key_store, new Param_pair("key_store", 6));
                put(Connection_enums.key_store_pwd, new Param_pair("key_store_pwd", 7));
                put(Connection_enums.user, new Param_pair("user", 100));
                put(Connection_enums.user_passwd, new Param_pair("password", 101));
              }
            };
    protected boolean parm_warning = false;
    public Connection_properties(Properties properties, String connection_url) {
      super();
      String[] url_values = connection_url.split(":");

      // Look for all possible properties values
      for (Connection_enums enum_key : connection_map.keySet()) {
        // Get each entry - a string to index the properties param (such as host_name
        // and an int to index it into the URL, such as 5 for host_name.
        // index will be greater than 99 if the value shouldn't be expected in the URL
        Param_pair param_pair = connection_map.get(enum_key);
        String value_from_prop = null;
        String value_from_url = null;
        // if the index is inside the range of the URL then grab the value
        if (param_pair.index < url_values.length) {
          value_from_url = url_values[param_pair.index];
        }

        // Grab the possible value from the properties variable, using the already
        // obtained value_from_url as the default if the entry isn't in properties
        // (note value_from_url could still be null in which case value_from_prop will
        // be null)
        value_from_prop = properties.getProperty(param_pair.label, value_from_url);
        if (value_from_url != null && value_from_prop != null) {
          if (!value_from_prop.equals(value_from_url)) {
            logger.warn("Connected property in url[" + value_from_url
                    + "] differs from Properties class [" + value_from_prop
                    + "]. Using url version");
            value_from_prop = value_from_url;
            parm_warning = true;
          }
        }
        if (value_from_prop != null) this.put(enum_key, value_from_prop);
      }
      // Make sure we have all that is needed and in the correct format
      validate_params();
    }
    private void validate_params() {
      // if present remove "//" from front of hostname
      String hN = (String) this.get(Connection_enums.host_name);
      if (hN.startsWith("//")) {
        this.put(Connection_enums.host_name, hN.substring(2));
      }
      Integer port_num = Integer.parseInt((String) (this.get(Connection_enums.port_num)));
      this.put(Connection_enums.port_num, port_num);
      String protocol = "binary";
      if (this.containsKey(Connection_enums.protocol)) {
        protocol = (String) this.get(Connection_enums.protocol);
        protocol.toLowerCase();
        if (!protocol.equals("binary") && !protocol.equals("http")
                && !protocol.equals("https")) {
          logger.warn("Incorrect protcol [" + protocol
                  + "] supplied. Possible values are [binary | http | https]. Using binary as default");
          protocol = "binary";
          parm_warning = true;
        }
        this.put(Connection_enums.protocol, protocol);
      }
      if (this.containsKey(Connection_enums.key_store)
              && !this.containsKey(Connection_enums.key_store_pwd)) {
        logger.warn("key store [" + (String) this.get(Connection_enums.key_store)
                + " specfied without a password");
        parm_warning = true;
      }
      if (this.containsKey(Connection_enums.key_store_pwd)
              && !this.containsKey(Connection_enums.key_store)) {
        logger.warn("key store password specified without a keystore file");
        parm_warning = true;
      }
    }

    boolean isHttpProtocol() {
      return (this.containsKey(Connection_enums.protocol)
              && this.get(Connection_enums.protocol).equals("http"));
    }
    boolean isHttpsProtocol() {
      return (this.containsKey(Connection_enums.protocol)
              && this.get(Connection_enums.protocol).equals("https"));
    }
    boolean isBinary() {
      return (this.containsKey(Connection_enums.protocol)
              && this.get(Connection_enums.protocol).equals("binary"));
    }
    boolean containsTrustStore() {
      return this.containsKey(Connection_enums.key_store);
    }
  } /* End class Connection_properties extends Hashtable<Connection_enums, Object> */

  protected String session = null;
  protected MapD.Client client = null;
  protected String catalog;
  protected TTransport transport;
  protected SQLWarning warnings;
  protected String url;
  protected Connection_properties cP = null;
  public OmniSciConnection(String url, Properties info)
          throws SQLException { // logger.debug("Entered");
    this.url = url;
    this.cP = new Connection_properties(info, url);
    SockTransportProperties skT = null;
    try {
      if (this.cP.containsTrustStore()) {
        skT = new SockTransportProperties(
                (String) this.cP.get(Connection_enums.key_store),
                (String) cP.get(Connection_enums.key_store_pwd));
      }
      TProtocol protocol = null;
      if (this.cP.isHttpProtocol()) {
        transport = SockTransportProperties.openHttpClientTransport(
                (String) this.cP.get(Connection_enums.host_name),
                ((Integer) this.cP.get(Connection_enums.port_num)).intValue(),
                skT);
        transport.open();
        protocol = new TJSONProtocol(transport);
      } else if (this.cP.isHttpsProtocol()) {
        transport = SockTransportProperties.openHttpsClientTransport(
                (String) this.cP.get(Connection_enums.host_name),
                ((Integer) this.cP.get(Connection_enums.port_num)).intValue(),
                skT);
        transport.open();
        protocol = new TJSONProtocol(transport);
      } else {
        transport = SockTransportProperties.openClientTransport(
                (String) this.cP.get(Connection_enums.host_name),
                ((Integer) this.cP.get(Connection_enums.port_num)).intValue(),
                skT);
        if (!transport.isOpen()) transport.open();
        protocol = new TBinaryProtocol(transport);
      }
      client = new MapD.Client(protocol);
      session = client.connect((String) this.cP.get(Connection_enums.user),
              (String) this.cP.get(Connection_enums.user_passwd),
              (String) this.cP.get(Connection_enums.db_name));

      logger.debug("Connected session is " + session);

    } catch (TTransportException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    } catch (TMapDException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    } catch (TException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    } catch (java.lang.Exception ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    }
  }

  @Override
  public Statement createStatement() throws SQLException { // logger.debug("Entered");
    return new OmniSciStatement(session, client);
  }

  @Override
  public PreparedStatement prepareStatement(String sql)
          throws SQLException { // logger.debug("Entered");
    return new OmniSciPreparedStatement(sql, session, client);
  }

  @Override
  public CallableStatement prepareCall(String sql)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public String nativeSQL(String sql) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setAutoCommit(boolean autoCommit)
          throws SQLException { // logger.debug("Entered");
    // we always autocommit per statement
  }

  @Override
  public boolean getAutoCommit() throws SQLException { // logger.debug("Entered");
    return true;
  }

  @Override
  public void commit() throws SQLException { // logger.debug("Entered");
    // noop
  }

  @Override
  public void rollback() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void close() throws SQLException { // logger.debug("Entered");
    try {
      logger.debug("Session at close is " + session);
      if (session != null) {
        client.disconnect(session);
      }
      session = null;
      transport.close();
    } catch (TMapDException ex) {
      throw new SQLException("disconnect failed." + ex.toString());
    } catch (TException ex) {
      throw new SQLException("disconnect failed." + ex.toString());
    }
  }

  @Override
  public boolean isClosed() throws SQLException { // logger.debug("Entered");
    if (session == null) {
      return true;
    }
    return false;
  }

  @Override
  public DatabaseMetaData getMetaData() throws SQLException { // logger.debug("Entered");
    DatabaseMetaData mapDMetaData = new OmniSciDatabaseMetaData(this);

    return mapDMetaData;
  }

  @Override
  public void setReadOnly(boolean readOnly)
          throws SQLException { // logger.debug("Entered");
    // TODO MAT we can't push the readonly upstream currently
    // but we could make JDBC obey this command
  }

  @Override
  public boolean isReadOnly() throws SQLException { // logger.debug("Entered");
    try {
      if (session != null) {
        TServerStatus server_status = client.get_server_status(session);
        return server_status.read_only;
      }
    } catch (TMapDException ex) {
      throw new SQLException(
              "get_server_status failed during isReadOnly check." + ex.toString());
    } catch (TException ex) {
      throw new SQLException(
              "get_server_status failed during isReadOnly check." + ex.toString());
    }
    // never should get here
    return true;
  }

  @Override
  public void setCatalog(String catalog) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public String getCatalog() throws SQLException { // logger.debug("Entered");
    return catalog;
  }

  @Override
  public void setTransactionIsolation(int level)
          throws SQLException { // logger.debug("Entered");
  }

  @Override
  public int getTransactionIsolation() throws SQLException { // logger.debug("Entered");
    return Connection.TRANSACTION_NONE;
  }

  @Override
  public SQLWarning getWarnings() throws SQLException { // logger.debug("Entered");
    return warnings;
  }

  @Override
  public void clearWarnings() throws SQLException { // logger.debug("Entered");
    warnings = null;
  }

  @Override
  public Statement createStatement(int resultSetType, int resultSetConcurrency)
          throws SQLException { // logger.debug("Entered");
    return new OmniSciStatement(session, client);
  }

  @Override
  public PreparedStatement prepareStatement(
          String sql, int resultSetType, int resultSetConcurrency)
          throws SQLException { // logger.debug("Entered");
    return new OmniSciPreparedStatement(sql, session, client);
  }

  @Override
  public CallableStatement prepareCall(
          String sql, int resultSetType, int resultSetConcurrency)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Map<String, Class<?>> getTypeMap()
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setTypeMap(Map<String, Class<?>> map)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setHoldability(int holdability)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getHoldability() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Savepoint setSavepoint() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Savepoint setSavepoint(String name)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void rollback(Savepoint savepoint)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void releaseSavepoint(Savepoint savepoint)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Statement createStatement(
          int resultSetType, int resultSetConcurrency, int resultSetHoldability)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public PreparedStatement prepareStatement(String sql,
          int resultSetType,
          int resultSetConcurrency,
          int resultSetHoldability) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public CallableStatement prepareCall(String sql,
          int resultSetType,
          int resultSetConcurrency,
          int resultSetHoldability) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public PreparedStatement prepareStatement(String sql, int autoGeneratedKeys)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public PreparedStatement prepareStatement(String sql, int[] columnIndexes)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public PreparedStatement prepareStatement(String sql, String[] columnNames)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Clob createClob() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Blob createBlob() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public NClob createNClob() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLXML createSQLXML() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isValid(int timeout) throws SQLException { // logger.debug("Entered");
    try {
      client.get_server_status(session);
    } catch (TTransportException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    } catch (TMapDException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    } catch (TException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    }
    return true;
  }

  @Override
  public void setClientInfo(String name, String value) throws SQLClientInfoException {
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClientInfo(Properties properties) throws SQLClientInfoException {
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public String getClientInfo(String name)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Properties getClientInfo() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Array createArrayOf(String typeName, Object[] elements)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Struct createStruct(String typeName, Object[] attributes)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setSchema(String schema) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public String getSchema() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void abort(Executor executor) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNetworkTimeout(Executor executor, int milliseconds)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public int getNetworkTimeout() throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface)
          throws SQLException { // logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }
}
