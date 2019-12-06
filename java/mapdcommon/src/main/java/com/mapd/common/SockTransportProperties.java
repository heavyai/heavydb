package com.mapd.common;

import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.conn.ssl.SSLSocketFactory;
import org.apache.http.conn.ssl.X509HostnameVerifier;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.ssl.SSLContexts;
import org.apache.thrift.transport.THttpClient;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.security.KeyStore;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.Arrays;

import javax.net.ssl.KeyManager;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import javax.net.ssl.X509TrustManager;

public class SockTransportProperties {
  final static org.slf4j.Logger MAPDLOGGER =
          LoggerFactory.getLogger(SockTransportProperties.class);

  /*
   * The following static 'factory' methods have been added to make usage of this
   * class simpler. The previously defined constructors have been left public to
   * limit the scope of the current change, though ideally they should be removed
   * and the calling code replaced with use of these static methods.
   */

  /**  Unencrypted Client method */
  static public SockTransportProperties getUnencryptedClient() throws Exception {
    boolean validate_server_name = false;
    return new SockTransportProperties(
            TransportType.unencryptedClient, validate_server_name);
  }
  /**  Encrypted Client method */
  static public SockTransportProperties getEncryptedClientDefaultTrustStore(
          boolean validate_server_name) throws Exception {
    return new SockTransportProperties(
            TransportType.encryptedClientDefaultTrustStore, validate_server_name);
  }

  // TODO for simplicity this method should be removed
  static public SockTransportProperties getEncryptedClientSpecifiedTrustStore(
          String trustStoreName, String trustStorePassword) throws Exception {
    return getEncryptedClientSpecifiedTrustStore(
            trustStoreName, trustStorePassword, true);
  }
  static public SockTransportProperties getEncryptedClientSpecifiedTrustStore(
          String trustStoreName, String trustStorePassword, boolean validate_server_name)
          throws Exception {
    return new SockTransportProperties(TransportType.encryptedClientSpecifiedTrustStore,
            trustStoreName,
            trustStorePassword,
            validate_server_name);
  }

  /** Server methods */
  static public SockTransportProperties getEncryptedServer(
          String keyStoreName, String keyStorePassword) throws Exception {
    boolean validate_server_name = false;
    if (keyStoreName == null || keyStorePassword == null) {
      String errStr = new String(
              "Invalid null parameter(s) used for getEncryptedServer. Both keyStoreName and keyStorePassword must be specified");
      RuntimeException rE = new RuntimeException(errStr);
      MAPDLOGGER.error(errStr, rE);
      throw(rE);
    }
    return new SockTransportProperties(TransportType.encryptedServer,
            keyStoreName,
            keyStorePassword,
            validate_server_name);
  }

  static public SockTransportProperties getUnecryptedServer() throws Exception {
    boolean validate_server_name = false;
    return new SockTransportProperties(
            TransportType.unencryptedServer, validate_server_name);
  }
  /**  End static method */

  // There are nominally 5 different 'types' of this class.
  private enum TransportType {
    encryptedServer,
    unencryptedServer,
    unencryptedClient,
    encryptedClientDefaultTrustStore,
    encryptedClientSpecifiedTrustStore
  }

  /** public constructor (for backward compatibility) */
  public SockTransportProperties(String truststore_name, String truststore_passwd)
          throws Exception {
    this(TransportType.encryptedClientSpecifiedTrustStore,
            truststore_name,
            truststore_passwd,
            true);
  }

  // TODO for simplicity this constructor should be removed
  public SockTransportProperties(boolean validate_server_name) throws Exception {
    this(TransportType.encryptedClientDefaultTrustStore, validate_server_name);
  }

  /** private constructors called from public static methods */
  private SockTransportProperties(TransportType tT,
          String store_name,
          String passwd,
          boolean validate_server_name) throws Exception {
    x509HostnameVerifier_ = (validate_server_name == true)
            ? SSLConnectionSocketFactory.STRICT_HOSTNAME_VERIFIER
            : SSLConnectionSocketFactory.ALLOW_ALL_HOSTNAME_VERIFIER;
    transportType = tT;

    char[] store_password = "".toCharArray();
    if (passwd != null && !passwd.isEmpty()) {
      store_password = passwd.toCharArray();
    }
    switch (transportType) {
      case encryptedServer: {
        key_store_password = store_password;
        key_store_name = store_name;
        break;
      }
      case encryptedClientSpecifiedTrustStore: {
        if (store_name == null) {
          initializeAcceptedIssuers(null);
        } else {
          KeyStore trust_store = KeyStore.getInstance(KeyStore.getDefaultType());
          try {
            java.io.FileInputStream fis = new java.io.FileInputStream(store_name);
            trust_store.load(fis, store_password);
          } catch (Exception eX) {
            String err_str =
                    new String("Error loading key/trust store [" + store_name + "]");
            MAPDLOGGER.error(err_str, eX);
            throw(eX);
          }
          initializeAcceptedIssuers(trust_store);
        }
        break;
      }
      default: {
        String errStr = new String(
                "Invalid transportType [" + transportType + "] used in constructor");
        RuntimeException rE = new RuntimeException(errStr);
        MAPDLOGGER.error(errStr, rE);
        throw(rE);
      }
    }
  }

  private SockTransportProperties(
          TransportType transportType, boolean validate_server_name) throws Exception {
    x509HostnameVerifier_ = (validate_server_name == true)
            ? SSLConnectionSocketFactory.STRICT_HOSTNAME_VERIFIER
            : SSLConnectionSocketFactory.ALLOW_ALL_HOSTNAME_VERIFIER;
    this.transportType = transportType;
    switch (transportType) {
      case encryptedClientDefaultTrustStore:
        // load default trust_store
        initializeAcceptedIssuers((KeyStore) null);
        break;
      case unencryptedClient:
      case unencryptedServer:
        break;
      default:
        String errStr = new String(
                "Invalid transportType [" + transportType + "] used in constructor");
        RuntimeException rE = new RuntimeException(errStr);
        MAPDLOGGER.error(errStr, rE);
        throw(rE);
    }
  }
  /** end private constructors  */

  private void initializeAcceptedIssuers(KeyStore trust_store) throws Exception {
    // Initialize a trust manager to either the  trust store already loaded or the
    // default trust store. Order of searching for  default is:
    // 1. system property javax.net.ssl.trustStore
    // 2. <java-home>/lib/security/jssecacerts
    // 3. <java-home</lib/security/cacerts

    TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance("PKIX");
    // If trust_store is null init will load the default trust_store
    trustManagerFactory.init(trust_store);
    trustManagers = trustManagerFactory.getTrustManagers();
  }

  /*
   * public client open transport methods
   *
   * openClientTransport opensHttpClientTransport, openHttpsClientTransport
   *
   */

  public TTransport openClientTransport(String server_host, int port)
          throws org.apache.thrift.TException {
    TTransport tTransport = null;
    switch (transportType) {
      case encryptedClientDefaultTrustStore:
      case encryptedClientSpecifiedTrustStore:
        tTransport = openBinaryEncrypted(server_host, port);
        break;
      case unencryptedClient:
        tTransport = new TSocket(server_host, port);
        break;
      default:
        String errStr = new String("Invalid transportType [" + transportType
                + "] used in openClientTransport");
        RuntimeException rE = new RuntimeException(errStr);
        MAPDLOGGER.error(errStr, rE);
        throw(rE);
    }
    return tTransport;
  }

  private TTransport openBinaryEncrypted(String server_host, int port)
          throws org.apache.thrift.TException {
    // Used to set Socket.setSoTimeout ms. 0 == inifinite.
    int socket_so_timeout_ms = 0;
    TSocket tsocket = null;
    try {
      SSLContext sc = SSLContext.getInstance("TLS");
      sc.init(null, trustManagers, new java.security.SecureRandom());

      SSLSocket sx = (SSLSocket) sc.getSocketFactory().createSocket(server_host, port);
      sx.setSoTimeout(socket_so_timeout_ms);
      tsocket = new TSocket(sx);
    } catch (Exception ex) {
      String errStr = new String("Error openBinaryEncrypted [" + server_host + ":" + port
              + "] used in openClientTransport - ");
      errStr += ex.toString();
      RuntimeException rE = new RuntimeException(errStr);
      MAPDLOGGER.error(errStr, rE);
      throw(rE);
    }
    return tsocket;
  }

  public TTransport openHttpsClientTransport(String server_host, int port)
          throws Exception {
    if (transportType != TransportType.encryptedClientDefaultTrustStore
            && transportType != TransportType.encryptedClientSpecifiedTrustStore) {
      String errStr = new String("Invalid transportType [" + transportType
              + "] used in openHttpsClientTransport");
      RuntimeException rE = new RuntimeException(errStr);
      MAPDLOGGER.error(errStr, rE);
      throw(rE);
    }
    TTransport transport = null;

    try {
      SSLContext sc = SSLContext.getInstance("TLS");
      sc.init(null, trustManagers, new java.security.SecureRandom());
      SSLConnectionSocketFactory sslConnectionSocketFactory = null;
      sslConnectionSocketFactory =
              new SSLConnectionSocketFactory(sc, x509HostnameVerifier_);

      CloseableHttpClient closeableHttpClient =
              HttpClients.custom()
                      .setSSLSocketFactory(sslConnectionSocketFactory)
                      .build();
      transport =
              new THttpClient("https://" + server_host + ":" + port, closeableHttpClient);

    } catch (Exception ex) {
      String err_str = new String("Exception:" + ex.getClass().getCanonicalName()
              + " thrown. Unable to create Secure socket for the HTTPS connection");
      MAPDLOGGER.error(err_str, ex);
      throw ex;
    }

    return transport;
  }

  public TTransport openHttpClientTransport(String server_host, int port)
          throws org.apache.thrift.TException {
    if (transportType != TransportType.unencryptedClient) {
      String errStr = new String("Invalid transportType [" + transportType
              + "] used in openHttpClientTransport");
      RuntimeException rE = new RuntimeException(errStr);
      MAPDLOGGER.error(errStr, rE);
      throw(rE);
    }

    String url = "http://" + server_host + ":" + port;
    return (new THttpClient(url));
  }

  /*
   * open Binary Server transport ***
   */
  public TServerTransport openServerTransport(int port)
          throws org.apache.thrift.TException {
    if (transportType == TransportType.encryptedServer) {
      return openServerTransportEncrypted(port);
    } else if (transportType == TransportType.unencryptedServer) {
      return (new TServerSocket(port));
    } else {
      String errStr = new String("Invalid transportType [" + transportType
              + "] used in openServerTransport");
      RuntimeException rE = new RuntimeException(errStr);
      MAPDLOGGER.error(errStr, rE);
      throw(rE);
    }
  }

  private TServerTransport openServerTransportEncrypted(int port)
          throws org.apache.thrift.TException {
    // Used to set Socket.setSoTimeout ms. 0 == inifinite.
    int socket_so_timeout_ms = 0;
    TSSLTransportFactory.TSSLTransportParameters params =
            new TSSLTransportFactory.TSSLTransportParameters();
    params.setKeyStore(key_store_name,
            (key_store_password != null) ? new String(key_store_password) : null);
    params.requireClientAuth(false);

    // return TSSLTransportFactory.getClientSocket(server_host, port,
    // socket_so_timeout_ms, params);
    TServerTransport t = TSSLTransportFactory.getServerSocket(
            port, socket_so_timeout_ms, null, params);
    return t;
  }

  private TrustManager[] trustManagers;
  private TransportType transportType = null;
  private KeyManager[] keyManagers;
  private String key_store_name = null;
  private char[] key_store_password = null;
  X509HostnameVerifier x509HostnameVerifier_ =
          SSLConnectionSocketFactory.BROWSER_COMPATIBLE_HOSTNAME_VERIFIER;
}
