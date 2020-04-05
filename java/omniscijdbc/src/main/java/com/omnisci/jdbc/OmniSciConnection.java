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

import com.mapd.common.SockTransportProperties;
import com.mapd.thrift.server.OmniSci;
import com.mapd.thrift.server.TOmniSciException;
import com.mapd.thrift.server.TServerStatus;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.security.*;
import java.security.cert.X509Certificate;
import java.security.spec.PKCS8EncodedKeySpec;
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

import javax.crypto.Cipher;

import sun.security.provider.X509Factory;
/**
 *
 * @author michael
 */
/*
 * Param_pair - Simple pair class to hold the label for a component in the url
 * and an index into the url to that component. For example in the url
 * jdbc:mapd:hostname:6278 a Param_pair for hostname would have a label of
 * "hostname" and an index of 2
 *
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
  server_trust_store,
  server_trust_store_pwd,
  pkiauth,
  sslcert,
  sslkey,
  sslkey_password,
  user,
  user_passwd
}

class KeyLoader {
  static class S_struct {
    public String cert;
    public Key key;
  }

  public static String getX509(X509Certificate cert) throws Exception {
    String encoded = Base64.getMimeEncoder().encodeToString(cert.getEncoded());
    // Note mimeEncoder inserts \r\n in the text - the server is okay with that.
    encoded = X509Factory.BEGIN_CERT + "\n" + encoded + "\n" + X509Factory.END_CERT;
    return encoded;
  }

  public static S_struct getDetails_pkcs12(String filename, String password)
          throws Exception {
    S_struct s_struct = new S_struct();
    try {
      KeyStore keystore = KeyStore.getInstance("PKCS12");
      java.io.FileInputStream fis = new java.io.FileInputStream(filename);
      keystore.load(fis, password.toCharArray());
      String alias = null;
      Enumeration<String> eE = keystore.aliases();
      int count = 0;
      while (eE.hasMoreElements()) {
        alias = eE.nextElement();
        count++;
      }
      if (count != 1) {
        throw new RuntimeException("pkcs12 file [" + filename
                + "] contains an incorrect number [" + count
                + "] of certificate(s); only a single certificate is allowed");
      }

      X509Certificate cert = (X509Certificate) keystore.getCertificate(alias);
      s_struct.cert = getX509(cert);
      s_struct.key = keystore.getKey(alias, password.toCharArray());
    } catch (Exception eX) {
      OmniSciConnection.logger.error(eX.getMessage());
      throw eX;
    }
    return s_struct;
  }
}

public class OmniSciConnection implements java.sql.Connection, Cloneable {
  final static Logger logger = LoggerFactory.getLogger(OmniSciConnection.class);

  Set<String> protocol_set = new HashSet<String>(
          Arrays.asList("binary", "binary_tls", "http", "https", "https_insecure"));
  // A simple internal class to hold a summary of the properties passed to the
  // connection
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
                put(Connection_enums.server_trust_store,
                        new Param_pair("server_trust_store", 6));
                put(Connection_enums.server_trust_store_pwd,
                        new Param_pair("server_trust_store_pwd", 7));
                put(Connection_enums.pkiauth, new Param_pair("pkiauth", 7));
                put(Connection_enums.sslcert, new Param_pair("sslcert", 8));
                put(Connection_enums.sslkey, new Param_pair("sslkey", 9));
                put(Connection_enums.sslkey_password,
                        new Param_pair("sslkey_password", 10));
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
      // Default to binary of no protocol specified
      String protocol = "binary";
      if (this.containsKey(Connection_enums.protocol)) {
        protocol = (String) this.get(Connection_enums.protocol);
        protocol.toLowerCase();
        if (!protocol_set.contains(protocol)) {
          logger.warn("Incorrect protcol [" + protocol
                  + "] supplied. Possible values are [" + protocol_set.toString()
                  + "]. Using binary as default");
          protocol = "binary";
          parm_warning = true;
        }
      }
      this.put(Connection_enums.protocol, protocol);
      if (this.containsKey(Connection_enums.server_trust_store)
              && !this.containsKey(Connection_enums.server_trust_store_pwd)) {
        logger.warn("server trust store ["
                + (String) this.get(Connection_enums.server_trust_store)
                + " specfied without a password");
        parm_warning = true;
      }
      if (this.containsKey(Connection_enums.server_trust_store_pwd)
              && !this.containsKey(Connection_enums.server_trust_store)) {
        logger.warn("server trust store password specified without a keystore file");
        parm_warning = true;
      }
    }

    boolean isHttpProtocol() {
      return (this.containsKey(Connection_enums.protocol)
              && this.get(Connection_enums.protocol).equals("http"));
    }

    boolean isHttpsProtocol_insecure() {
      return (this.containsKey(Connection_enums.protocol)
              && this.get(Connection_enums.protocol).equals("https_insecure"));
    }

    boolean isHttpsProtocol() {
      return (this.containsKey(Connection_enums.protocol)
              && this.get(Connection_enums.protocol).equals("https"));
    }

    boolean isBinary() {
      return (this.containsKey(Connection_enums.protocol)
              && this.get(Connection_enums.protocol).equals("binary"));
    }
    boolean isBinary_tls() {
      return (this.containsKey(Connection_enums.protocol)
              && this.get(Connection_enums.protocol).equals("binary_tls"));
    }
    boolean containsTrustStore() {
      return this.containsKey(Connection_enums.server_trust_store);
    }
  } /*
     * End class Connection_properties extends Hashtable<Connection_enums, Object>
     */

  protected String session = null;
  protected OmniSci.Client client = null;
  protected String catalog;
  protected TTransport transport;
  protected SQLWarning warnings;
  protected String url;
  protected Connection_properties cP = null;

  public OmniSciConnection getAlternateConnection() throws SQLException {
    // Clones the orignal java connection object, and then reconnects
    // at the thrift layer - doesn't re-authenticate at the application
    // level.  Instead reuses the orignal connections session number.
    logger.debug("OmniSciConnection clone");
    OmniSciConnection omniSciConnection = null;
    try {
      omniSciConnection = (OmniSciConnection) super.clone();
    } catch (CloneNotSupportedException eE) {
      throw new SQLException(
              "Error cloning connection [" + OmniSciExceptionText.getExceptionDetail(eE),
              eE);
    }
    // Now over write the old connection.
    try {
      TProtocol protocol = omniSciConnection.manageConnection();
      omniSciConnection.client = new OmniSci.Client(protocol);
    } catch (java.lang.Exception jE) {
      throw new SQLException("Error creating new connection "
                      + OmniSciExceptionText.getExceptionDetail(jE),
              jE);
    }
    return omniSciConnection;
  }

  private TProtocol manageConnection() throws java.lang.Exception {
    SockTransportProperties skT = null;
    String trust_store = null;
    String trust_store_pwd = null;
    TProtocol protocol = null;
    if (this.cP.isHttpProtocol()) {
      // HTTP
      skT = SockTransportProperties.getUnencryptedClient();

      transport = skT.openHttpClientTransport(
              (String) this.cP.get(Connection_enums.host_name),
              ((Integer) this.cP.get(Connection_enums.port_num)).intValue());
      transport.open();
      protocol = new TJSONProtocol(transport);

    } else if (this.cP.isBinary()) {
      skT = SockTransportProperties.getUnencryptedClient();
      transport =
              skT.openClientTransport((String) this.cP.get(Connection_enums.host_name),
                      ((Integer) this.cP.get(Connection_enums.port_num)).intValue());
      if (!transport.isOpen()) transport.open();
      protocol = new TBinaryProtocol(transport);

    } else if (this.cP.isHttpsProtocol() || this.cP.isHttpsProtocol_insecure()) {
      if (trust_store == null) {
        skT = SockTransportProperties.getEncryptedClientDefaultTrustStore(
                !this.cP.isHttpsProtocol_insecure());
      } else {
        skT = SockTransportProperties.getEncryptedClientSpecifiedTrustStore(
                trust_store, trust_store_pwd, !this.cP.isHttpsProtocol_insecure());
      }
      transport = skT.openHttpsClientTransport(
              (String) this.cP.get(Connection_enums.host_name),
              ((Integer) this.cP.get(Connection_enums.port_num)).intValue());
      transport.open();
      protocol = new TJSONProtocol(transport);

    } else if (cP.isBinary_tls()) {
      if (trust_store == null) {
        skT = SockTransportProperties.getEncryptedClientDefaultTrustStore(false);
      } else {
        skT = SockTransportProperties.getEncryptedClientSpecifiedTrustStore(
                trust_store, trust_store_pwd, false);
      }
      transport =
              skT.openClientTransport((String) this.cP.get(Connection_enums.host_name),
                      ((Integer) this.cP.get(Connection_enums.port_num)).intValue());

      if (!transport.isOpen()) transport.open();
      protocol = new TBinaryProtocol(transport);
    } else {
      throw new RuntimeException("Invalid protocol supplied");
    }
    return protocol;
  }

  private void setSession(Object pki_auth) throws java.lang.Exception {
    KeyLoader.S_struct s_struct = null;
    // If pki aut then stuff public cert into password.
    if (pki_auth != null && pki_auth.toString().equalsIgnoreCase("true")) {
      s_struct = KeyLoader.getDetails_pkcs12(
              this.cP.get(Connection_enums.sslcert).toString(),
              this.cP.get(Connection_enums.sslkey_password).toString());
      this.cP.put(Connection_enums.user_passwd, s_struct.cert);
    }

    // Get the seesion for all connectioms
    session = client.connect((String) this.cP.get(Connection_enums.user),
            (String) this.cP.get(Connection_enums.user_passwd),
            (String) this.cP.get(Connection_enums.db_name));

    // if pki auth the session will be encoded.
    if (pki_auth != null && pki_auth.toString().equalsIgnoreCase("true")) {
      Cipher cipher = Cipher.getInstance(s_struct.key.getAlgorithm());
      cipher.init(Cipher.DECRYPT_MODE, s_struct.key);
      // session is encrypted and encoded in b64
      byte[] decodedBytes = Base64.getDecoder().decode(session);
      byte[] decoded_bytes = cipher.doFinal(decodedBytes);
      session = new String(decoded_bytes, "UTF-8");
    }
  }

  public OmniSciConnection(String url, Properties info)
          throws SQLException { // logger.debug("Entered");
    this.url = url;
    this.cP = new Connection_properties(info, url);

    try {
      TProtocol protocol = manageConnection();
      client = new OmniSci.Client(protocol);
      setSession(this.cP.get(Connection_enums.pkiauth));
      catalog = (String) this.cP.get(Connection_enums.db_name);
    } catch (TTransportException ex) {
      throw new SQLException("Thrift transport connection failed - "
                      + OmniSciExceptionText.getExceptionDetail(ex),
              ex);
    } catch (TOmniSciException ex) {
      throw new SQLException("Omnisci connection failed - "
                      + OmniSciExceptionText.getExceptionDetail(ex),
              ex);
    } catch (TException ex) {
      throw new SQLException(
              "Thrift failed - " + OmniSciExceptionText.getExceptionDetail(ex), ex);
    } catch (java.lang.Exception ex) {
      throw new SQLException(
              "Connection failed - " + OmniSciExceptionText.getExceptionDetail(ex), ex);
    }
  }

  @Override
  public Statement createStatement() throws SQLException { // logger.debug("Entered");
    return new OmniSciStatement(session, this);
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
      closeConnection();
    } catch (TOmniSciException ex) {
      throw new SQLException("disconnect failed." + ex.toString());
    } catch (TException ex) {
      throw new SQLException("disconnect failed." + ex.toString());
    }
  }

  // needs to be accessed by other classes with in the package
  protected void closeConnection() throws SQLException { // logger.debug("Entered");
    session = null;
    transport.close();
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
    } catch (TOmniSciException ex) {
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
    return new OmniSciStatement(session, this);
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
    } catch (TOmniSciException ex) {
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
    // Setting setSchema to be a NOOP allows integration with third party products
    // that require a successful call to setSchema to work.
    Object db_name = this.cP.get(Connection_enums.db_name);
    if (db_name == null) {
      throw new RuntimeException("db name not set, "
              + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
              + " class:" + new Throwable().getStackTrace()[0].getClassName()
              + " method:" + new Throwable().getStackTrace()[0].getMethodName());
    }
    if (!schema.equals(db_name.toString())) {
      logger.warn("Connected to schema [" + schema + "] differs from db name [" + db_name
              + "].");
    }
    return;
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
