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
package com.mapd.jdbc;

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
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.Executor;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.THttpClient;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class MapDConnection implements java.sql.Connection {

  final static Logger logger = LoggerFactory.getLogger(MapDConnection.class);

  protected String session = null;
  protected MapD.Client client = null;
  protected String url = null;
  protected Properties properties = null;
  protected String user;
  protected TTransport transport;
  protected SQLWarning warnings;

  public MapDConnection(String url, Properties info) throws SQLException { //logger.debug("Entered");
    this.url = url;
    this.properties = info;
    boolean http_session = false;

    //logger.debug("We got to here " + url + " info: " + info.toString());
    String[] temp = url.split(":");

    //for (int i = 0; i < temp.length; i++) {
    //  logger.debug("temp  " + i + " " + temp[i].toString());
    //}
    String machine = temp[2];

    // deal with requirement that there may be double // before the machine
    if (machine.startsWith("//")) {
      machine = machine.substring(2);
    }

    //logger.debug("machine : " + machine);
    int port = Integer.valueOf(temp[3]);
    String db = temp[4];
    //test for http protocol request (we could consider usinig properties)
    if (temp.length == 6) {
      if (temp[5].equals("http")) {
        http_session = true;
      } else {
        throw new SQLException("Connection failed invalid protocol option- " + temp[5]);
      }
    }
    try {
      TProtocol protocol = null;
      if (http_session) {
        transport = new THttpClient("http://" + machine + ":" + port);
        transport.open();
        protocol = new TJSONProtocol(transport);
      } else {
        transport = new TSocket(machine, port);
        transport.open();
        protocol = new TBinaryProtocol(transport);
      }
      client = new MapD.Client(protocol);

      session = client.connect(info.getProperty("user"), info.getProperty("password"), db);

      logger.debug("Connected session is " + session);

    } catch (TTransportException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    } catch (TMapDException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    } catch (TException ex) {
      throw new SQLException("Connection failed - " + ex.toString());
    }
  }

  @Override
  public Statement createStatement() throws SQLException { //logger.debug("Entered");
    return new MapDStatement(session, client);
  }

  @Override
  public PreparedStatement prepareStatement(String sql) throws SQLException { //logger.debug("Entered");
    return new MapDPreparedStatement(sql, session, client);
  }

  @Override
  public CallableStatement prepareCall(String sql) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public String nativeSQL(String sql) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setAutoCommit(boolean autoCommit) throws SQLException { //logger.debug("Entered");
    // we always autocommit per statement
  }

  @Override
  public boolean getAutoCommit() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public void commit() throws SQLException { //logger.debug("Entered");
    //noop
  }

  @Override
  public void rollback() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void close() throws SQLException { //logger.debug("Entered");
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
  public boolean isClosed() throws SQLException { //logger.debug("Entered");
    if (session == null) {
      return true;
    }
    return false;
  }

  @Override
  public DatabaseMetaData getMetaData() throws SQLException { //logger.debug("Entered");
    DatabaseMetaData mapDMetaData = new MapDDatabaseMetaData(this);

    return mapDMetaData;
  }

  @Override
  public void setReadOnly(boolean readOnly) throws SQLException { //logger.debug("Entered");
    // TODO MAT we can't push the readonly upstream currently 
    // but we could make JDBC obey this command
  }

  @Override
  public boolean isReadOnly() throws SQLException { //logger.debug("Entered");
    try {
      if (session != null) {
        TServerStatus server_status = client.get_server_status(session);
        return server_status.read_only;
      }
    } catch (TMapDException ex) {
      throw new SQLException("get_server_status failed during isReadOnly check." + ex.toString());
    } catch (TException ex) {
      throw new SQLException("get_server_status failed during isReadOnly check." + ex.toString());
    }
    // never should get here
    return true;
  }

  @Override
  public void setCatalog(String catalog) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public String getCatalog() throws SQLException { //logger.debug("Entered");
    return user;
  }

  @Override
  public void setTransactionIsolation(int level) throws SQLException { //logger.debug("Entered");
  }

  @Override
  public int getTransactionIsolation() throws SQLException { //logger.debug("Entered");
    return Connection.TRANSACTION_NONE;
  }

  @Override
  public SQLWarning getWarnings() throws SQLException { //logger.debug("Entered");
    return warnings;
  }

  @Override
  public void clearWarnings() throws SQLException { //logger.debug("Entered");
    warnings = null;
  }

  @Override
  public Statement createStatement(int resultSetType, int resultSetConcurrency) throws SQLException { //logger.debug("Entered");
    return new MapDStatement(session, client);
  }

  @Override
  public PreparedStatement prepareStatement(String sql, int resultSetType, int resultSetConcurrency) throws SQLException { //logger.debug("Entered");
    return new MapDPreparedStatement(sql, session, client);
  }

  @Override
  public CallableStatement prepareCall(String sql, int resultSetType, int resultSetConcurrency) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Map<String, Class<?>> getTypeMap() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setTypeMap(Map<String, Class<?>> map) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setHoldability(int holdability) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getHoldability() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Savepoint setSavepoint() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Savepoint setSavepoint(String name) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void rollback(Savepoint savepoint) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void releaseSavepoint(Savepoint savepoint) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Statement createStatement(int resultSetType, int resultSetConcurrency, int resultSetHoldability) throws
          SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public PreparedStatement prepareStatement(String sql, int resultSetType, int resultSetConcurrency,
          int resultSetHoldability) throws
          SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public CallableStatement prepareCall(String sql, int resultSetType, int resultSetConcurrency, int resultSetHoldability)
          throws
          SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public PreparedStatement prepareStatement(String sql, int autoGeneratedKeys) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public PreparedStatement prepareStatement(String sql, int[] columnIndexes) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public PreparedStatement prepareStatement(String sql, String[] columnNames) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Clob createClob() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Blob createBlob() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public NClob createNClob() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public SQLXML createSQLXML() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isValid(int timeout) throws SQLException { //logger.debug("Entered");
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
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setClientInfo(Properties properties) throws SQLClientInfoException {
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public String getClientInfo(String name) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Properties getClientInfo() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Array createArrayOf(String typeName, Object[] elements) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public Struct createStruct(String typeName, Object[] attributes) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setSchema(String schema) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public String getSchema() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void abort(Executor executor) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public void setNetworkTimeout(Executor executor, int milliseconds) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public int getNetworkTimeout() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

  @Override
  public boolean isWrapperFor(Class<?> iface) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet," + " line:" + new Throwable().getStackTrace()[0].
            getLineNumber() + " class:" + new Throwable().getStackTrace()[0].getClassName() + " method:" + new Throwable().
            getStackTrace()[0].getMethodName());
  }

}
