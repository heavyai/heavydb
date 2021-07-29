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
import com.omnisci.thrift.server.OmniSci;
import com.omnisci.thrift.server.TDatumType;
import com.omnisci.thrift.server.TOmniSciException;
import com.omnisci.thrift.server.TServerStatus;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.security.*;
import java.security.cert.X509Certificate;
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
        throw new SQLException("pkcs12 file [" + filename
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

class Options {
  // The Options class supplies the keys for the
  // Connection_properties class
  static String host_name = "host_name";
  static String port_num = "port_num";
  static String db_name = "db_name";
  static String protocol = "protocol";
  static String server_trust_store = "server_trust_store";
  static String server_trust_store_pwd = "server_trust_store_pwd";
  static String pkiauth = "pkiauth";
  static String sslcert = "sslcert";
  static String sslkey = "sslkey";
  static String sslkey_password = "sslkey_password";
  static String max_rows = "max_rows";
  static String user = "user";
  static String password = "password";
  // The order in the array corresponds to the order the value will appear in the ':'
  // separated URL. Used to loop over the incoming URL and store in the properties struct
  // Note user and password are not expected to come via the main body  of the url
  // However they may be supplied in the query string portion.

  static String[] option_order = {host_name,
          port_num,
          db_name,
          protocol,
          server_trust_store,
          server_trust_store_pwd,
          pkiauth,
          sslcert,
          sslkey,
          sslkey_password,
          max_rows};
}

public class OmniSciConnection implements java.sql.Connection, Cloneable {
  final static Logger logger = LoggerFactory.getLogger(OmniSciConnection.class);

  Set<String> protocol_set = new HashSet<String>(
          Arrays.asList("binary", "binary_tls", "http", "https", "https_insecure"));

  class Connection_properties extends Properties {
    // Properties can come in three ways:
    // 1. via main part of the url,
    // 2, via the query string portion of the URL or
    // 3. via a Properties param
    //
    // Example url. Note the first two fields are constants and must be present. '
    // jdbc:omnisci:localhost:4247:db_name?max_rows=2000&protocol=binary
    //
    // Priority is given to the URL data, followed by the query fragment data and
    // lastly the Properties information
    //
    // All connection parameters can be supplied via the main portion of the connection
    // URL, however as this structure is position dependent the nth argument must be
    // preceded by the n - 1  arguments.  In the case of complex connection strings it is
    // recommended to use either a properties object or the query string portion of the
    // URL.
    //
    // Note the class java.sql.DriverManager expects the URL to contain three components
    // the literal string JDBC a 'subprotocol' followed by a 'subname', as in
    // 'jdbc:omnisci:localhost' For this reason host mame must be supplied in the  main
    // part of the URL and should not be supplied in the query portion.

    private String extract_and_remove_query_components(
            String connection_url, Properties query_props) throws SQLException {
      // The omnisci version of the connection_url is a ':' separated list
      // with an optional 'query component' at the end (see example above).
      // The query component starts with the first '?' in the string.
      // Its is made up of key=value pairs separated by the '&' character.
      //
      // The query component is terminated by the end of the string and is
      // assumed to be at the end of the URL

      String[] url_components = connection_url.split("\\?");
      if (url_components.length == 2) {
        // Query component are separated by an '&' - replace each with a '\n'
        // will allow Properties.load method to build a properties obj
        StringReader reader = new StringReader(url_components[1].replace('&', '\n'));
        try {
          query_props.load(reader);
        } catch (IOException iex) {
          throw new SQLException(iex.toString());
        }
      } else if (url_components.length > 2) {
        throw new SQLException(
                "Invalid connection string. Multiple query components included ["
                + connection_url + "]");
      }
      // return the url with out any query component
      return url_components[0];
    }
    public Connection_properties(String connection_url, Properties baseProperties)
            throws SQLException {
      connection_url = extract_and_remove_query_components(connection_url, this);
      String[] url_values = connection_url.split(":");
      // add 2 for the default jdbc:omnisci at the start of the url.
      if (url_values.length > Options.option_order.length + 2) {
        // would be nice to print the url at this stage, but the user may have added their
        // password into the list.
        throw new SQLException("Invalid number of arguments provided in url ["
                + url_values.length + "]. Maximum allowed ["
                + (Options.option_order.length + 2) + "]");
      }
      for (int i = 2; i < url_values.length; i++) {
        // the offest of 2 is caused by the 2 lables 'jdbc:omnsci' at the start if the URL
        String existingValue = getProperty(Options.option_order[i - 2]);
        if (existingValue != null && !existingValue.equals((url_values[i]))) {
          logger.warn("Connection property [" + Options.option_order[i - 2]
                  + "] has been provided with different values in the URL and query component of the url. Defaulting to the URL value");
        }
        setProperty(Options.option_order[i - 2], url_values[i]);
      }

      for (String key : baseProperties.stringPropertyNames()) {
        String existingValue = getProperty(key);
        if (existingValue != null
                && !existingValue.equals(baseProperties.getProperty(key))) {
          logger.warn("Connection property " + key
                  + "] has been provided with different values in the properties object and the url. Defaulting to the URL value");
        } else {
          setProperty(key, baseProperties.getProperty(key));
        }
      }

      validate_params();
    }

    private void validate_params() throws SQLException {
      // Warn if config values with invalid keys have been used.
      for (String key : this.stringPropertyNames()) {
        if (key != Options.user && key != Options.password
                && !Arrays.asList(Options.option_order).contains(key)) {
          logger.warn("Unsupported configuration key" + key + " used.");
        }
      }
      // if present remove "//" from front of hostname
      if (containsKey(Options.host_name)) {
        String hN = this.getProperty(Options.host_name);
        if (hN.startsWith("//")) {
          this.setProperty(Options.host_name, hN.substring(2));
        }
      }

      // Default to binary if no protocol specified
      String protocol = "binary";
      if (this.containsKey(Options.protocol)) {
        protocol = this.getProperty(Options.protocol);
        protocol.toLowerCase();
        if (!protocol_set.contains(protocol)) {
          logger.warn("Incorrect protcol [" + protocol
                  + "] supplied. Possible values are [" + protocol_set.toString()
                  + "]. Using binary as default");
          protocol = "binary";
        }
      }
      this.setProperty(Options.protocol, protocol);

      if (this.containsKey(Options.port_num)) {
        try {
          Integer.parseInt(getProperty(Options.port_num));
        } catch (NumberFormatException nfe) {
          throw new SQLException(
                  "Invalid port number supplied" + getProperty(Options.port_num));
        }
      }

      if (this.containsKey(Options.server_trust_store)
              && !this.containsKey(Options.server_trust_store_pwd)) {
        logger.warn("server trust store ["
                + (String) this.getProperty(Options.server_trust_store)
                + " specfied without a password");
      }
      if (this.containsKey(Options.server_trust_store_pwd)
              && !this.containsKey(Options.server_trust_store)) {
        logger.warn("server trust store password specified without a keystore file");
      }
      if (!this.containsKey(Options.max_rows)) {
        this.setProperty(Options.max_rows, "100000");
      } else {
        try {
          Integer.parseInt(getProperty(Options.max_rows));
        } catch (NumberFormatException nfe) {
          throw new SQLException(
                  "Invalid value supplied for max rows " + getProperty(Options.max_rows));
        }
      }
    }

    boolean isHttpProtocol() {
      return (this.containsKey(Options.protocol)
              && this.getProperty(Options.protocol).equals("http"));
    }

    boolean isHttpsProtocol_insecure() {
      return (this.containsKey(Options.protocol)
              && this.getProperty(Options.protocol).equals("https_insecure"));
    }

    boolean isHttpsProtocol() {
      return (this.containsKey(Options.protocol)
              && this.getProperty(Options.protocol).equals("https"));
    }

    boolean isBinary() {
      return (this.containsKey(Options.protocol)
              && this.getProperty(Options.protocol).equals("binary"));
    }
    boolean isBinary_tls() {
      return (this.containsKey(Options.protocol)
              && this.getProperty(Options.protocol).equals("binary_tls"));
    }
    boolean containsTrustStore() {
      return this.containsKey(Options.server_trust_store);
    }
  }
  /*
   * End class Connection_properties
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

  // any java.lang.Exception thrown is caught downstream and converted
  // to a SQLException
  private TProtocol manageConnection() throws java.lang.Exception {
    SockTransportProperties skT = null;
    String trust_store = null;
    if (cP.getProperty(Options.server_trust_store) != null
            && !cP.getProperty(Options.server_trust_store).isEmpty()) {
      trust_store = cP.getProperty(Options.server_trust_store);
    }
    String trust_store_pwd = null;
    if (cP.getProperty(Options.server_trust_store_pwd) != null
            && !cP.getProperty(Options.server_trust_store_pwd).isEmpty()) {
      trust_store_pwd = cP.getProperty(Options.server_trust_store_pwd);
    }

    TProtocol protocol = null;
    if (this.cP.isHttpProtocol()) {
      // HTTP
      skT = SockTransportProperties.getUnencryptedClient();

      transport = skT.openHttpClientTransport(this.cP.getProperty(Options.host_name),
              Integer.parseInt(this.cP.getProperty(Options.port_num)));
      transport.open();
      protocol = new TJSONProtocol(transport);

    } else if (this.cP.isBinary()) {
      skT = SockTransportProperties.getUnencryptedClient();
      transport = skT.openClientTransport(this.cP.getProperty(Options.host_name),
              Integer.parseInt(this.cP.getProperty(Options.port_num)));
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
      transport = skT.openHttpsClientTransport(this.cP.getProperty(Options.host_name),
              Integer.parseInt(this.cP.getProperty(Options.port_num)));
      transport.open();
      protocol = new TJSONProtocol(transport);

    } else if (cP.isBinary_tls()) {
      if (trust_store == null) {
        skT = SockTransportProperties.getEncryptedClientDefaultTrustStore(false);
      } else {
        skT = SockTransportProperties.getEncryptedClientSpecifiedTrustStore(
                trust_store, trust_store_pwd, false);
      }
      transport = skT.openClientTransport(this.cP.getProperty(Options.host_name),
              Integer.parseInt(this.cP.getProperty(Options.port_num)));

      if (!transport.isOpen()) transport.open();
      protocol = new TBinaryProtocol(transport);
    } else {
      throw new SQLException("Invalid protocol supplied");
    }
    return protocol;
  }

  private void setSession(Object pki_auth) throws java.lang.Exception {
    KeyLoader.S_struct s_struct = null;
    // If pki aut then stuff public cert into password.
    if (pki_auth != null && pki_auth.toString().equalsIgnoreCase("true")) {
      s_struct = KeyLoader.getDetails_pkcs12(this.cP.getProperty(Options.sslcert),
              this.cP.getProperty(Options.sslkey_password));
      this.cP.setProperty(Options.password, s_struct.cert);
    }

    // Get the seesion for all connectioms
    session = client.connect((String) this.cP.getProperty(Options.user),
            (String) this.cP.getProperty(Options.password),
            (String) this.cP.getProperty(Options.db_name));

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

  public OmniSciConnection(String url, Properties base_properties) throws SQLException {
    this.url = url;
    this.cP = new Connection_properties(url, base_properties);
    try {
      TProtocol protocol = manageConnection();
      client = new OmniSci.Client(protocol);
      setSession(this.cP.getProperty(Options.pkiauth));
      catalog = (String) this.cP.getProperty(Options.db_name);
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
    return new OmniSciPreparedStatement(sql, session, this);
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
      throw new SQLException("disconnect failed." + ex.getError_msg());
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
              "get_server_status failed during isReadOnly check." + ex.getError_msg());
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
    return new OmniSciPreparedStatement(sql, session, this);
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
      throw new SQLException("Connection failed - " + ex.getError_msg());
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
    TDatumType type;
    try {
      type = TDatumType.valueOf(typeName.toUpperCase());
    } catch (IllegalArgumentException ex) {
      throw new SQLException("No matching omnisci type for " + typeName);
    }
    return new OmniSciArray(type, elements);
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
    Object db_name = this.cP.getProperty(Options.db_name);
    if (db_name == null) {
      throw new SQLException("db name not set, "
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
