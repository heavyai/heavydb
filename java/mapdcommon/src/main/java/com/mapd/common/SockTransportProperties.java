package com.mapd.common;

import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.ssl.SSLContexts;
import org.apache.thrift.transport.THttpClient;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.slf4j.LoggerFactory;

import java.security.KeyStore;
import java.security.cert.X509Certificate;

import javax.net.ssl.SSLContext;
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
  static public SockTransportProperties getUnencryptedClient() throws Exception {
    return new SockTransportProperties(TransportType.unencryptedClient);
  }

  static public SockTransportProperties getEncryptedClientPermisive() throws Exception {
    return new SockTransportProperties(TransportType.encryptedClientPermissive);
  }

  static public SockTransportProperties getEncryptedClientDefaultTrustStore()
          throws Exception {
    return new SockTransportProperties(TransportType.encryptedClientDefaultTrustStore);
  }

  static public SockTransportProperties getEncryptedClientSpecifiedTrustStore(
          String trustStoreName, String trustStorePassword) throws Exception {
    return new SockTransportProperties(TransportType.encryptedClientSpecifiedTrustStore,
            trustStoreName,
            trustStorePassword);
  }

  static public SockTransportProperties getEncryptedServer(
          String keyStoreName, String keyStorePassword) throws Exception {
    return new SockTransportProperties(
            TransportType.encryptedServer, keyStoreName, keyStorePassword);
  }

  static public SockTransportProperties getUnecryptedServer() throws Exception {
    return new SockTransportProperties(TransportType.unencryptedServer);
  }

  /* There are nominally 5 different 'types' of this class. */
  private enum TransportType {
    encryptedServer,
    unencryptedServer,
    unencryptedClient,
    encryptedClientPermissive,
    encryptedClientDefaultTrustStore,
    encryptedClientSpecifiedTrustStore
  }

  public SockTransportProperties(boolean load_trust) throws Exception {
    this((load_trust == true) ? TransportType.encryptedClientDefaultTrustStore
                              : TransportType.encryptedClientPermissive);
  }

  public SockTransportProperties(String truststore_name, String truststore_passwd)
          throws Exception {
    this(TransportType.encryptedClientSpecifiedTrustStore,
            truststore_name,
            truststore_passwd);
  }

  private SockTransportProperties(TransportType tT, String name, String passwd)
          throws Exception {
    transportType = tT;
    // Load the supplied file into a java keystore object to ensure it is okay. If
    // the keystore
    // contain client trust information it is used to initialize the TrustManager
    // factory, otherwise it is discarded.
    // The file name is stored in a member variable. When the required
    // open method is called the stored file name is passed to the appropriate
    // TSSLTransportParameters
    // method.
    KeyStore kS = KeyStore.getInstance(KeyStore.getDefaultType());
    char[] store_password = null;
    String store_name = null;
    if (passwd != null && !passwd.isEmpty()) {
      store_password = passwd.toCharArray();
    }
    if (name != null && !name.isEmpty()) {
      store_name = name;
    }
    try {
      java.io.FileInputStream fis = new java.io.FileInputStream(name);
      kS.load(fis, store_password);
    } catch (Exception eX) {
      String err_str = new String("Error loading key/trut store [" + name + "]");
      MAPDLOGGER.error(err_str, eX);
      throw(eX);
    }

    if (transportType == TransportType.encryptedServer) {
      // key_store_set = true;
      key_store_password = store_password;
      key_store_name = store_name;
    } else {
      initializeAcceptedIssuers(kS);
      trust_store_password = store_password;
      trust_store_name = store_name;
      // trust_store_set = true;
    }
  }

  private SockTransportProperties(TransportType transportType) throws Exception {
    // This constructor will either not bother loading trust data (and then trust
    // all server certs) or load from the java default trust stores.
    this.transportType = transportType;
    switch (transportType) {
      case encryptedClientDefaultTrustStore:
        initializeAcceptedIssuers((KeyStore) null);
        break;
      case encryptedClientPermissive:
        // trust_all_certs = true;
      case unencryptedClient:
        // do nothing
      case unencryptedServer:
        // do nothing
        break;
      default:
        String errStr = new String(
                "Invalid transportType [" + transportType + "] used in constructor");
        RuntimeException rE = new RuntimeException(errStr);
        MAPDLOGGER.error(errStr, rE);
        throw(rE);
    }
  }

  private void initializeAcceptedIssuers(KeyStore kS) throws Exception {
    TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance("PKIX");
    trustManagerFactory.init(kS);
    trustManagers = trustManagerFactory.getTrustManagers();
    X509TrustManager x509TrustManager = (X509TrustManager) trustManagers[0];
  }

  /*
   * open HTTPS *********************
   */

  private static X509TrustManager createInsecureTrustManager() {
    return new X509TrustManager() {
      public X509Certificate[] getAcceptedIssuers() {
        return null;
      }

      public void checkClientTrusted(X509Certificate[] certs, String authType) {}

      public void checkServerTrusted(X509Certificate[] certs, String authType) {}
    };
  }

  public TTransport openHttpsClientTransport(String server_host, int port)
          throws Exception {
    // Simple TrustManager to trust all certs
    TrustManager[] trustAllCerts = {createInsecureTrustManager()};

    TTransport transport = null;
    try {
      // Build a regular apache ClosableHttpClient based on a SSL connection
      // that can be passed to the apache thrift THttpClient constructor.

      SSLContext sc = null;
      sc = SSLContexts.custom().useProtocol("SSL").build();
      SSLConnectionSocketFactory sslConnectionSocketFactory = null;
      if (transportType == TransportType.encryptedClientPermissive) {
        sc.init(null, trustManagers, new java.security.SecureRandom());
        sslConnectionSocketFactory = new SSLConnectionSocketFactory(
                sc, SSLConnectionSocketFactory.BROWSER_COMPATIBLE_HOSTNAME_VERIFIER);
      } else {
        sc.init(null, trustAllCerts, new java.security.SecureRandom());
        sslConnectionSocketFactory = new SSLConnectionSocketFactory(
                sc, SSLConnectionSocketFactory.ALLOW_ALL_HOSTNAME_VERIFIER);
      }
      CloseableHttpClient closeableHttpClient =
              HttpClients.custom()
                      .setSSLSocketFactory(sslConnectionSocketFactory)
                      .build();
      transport =
              new THttpClient("https://" + server_host + ":" + port, closeableHttpClient);
    } catch (Exception ex) {
      String err_str = new String("Exception:" + ex.getClass().getCanonicalName()
              + " thown. Unable to create Secure socket for the HTTPS connection");
      MAPDLOGGER.error(err_str, ex);
      throw ex;
    }

    return transport;
  }

  /*
   * open HTTP *********************
   */
  public TTransport openHttpClientTransport(String server_host, int port)
          throws org.apache.thrift.TException {
    String url = "http://" + server_host + ":" + port;
    return (new THttpClient(url));
  }

  /*
   * open Binary Server *********************
   */
  public TServerTransport openServerTransport(int port)
          throws org.apache.thrift.TException {
    if (transportType == TransportType.encryptedServer) {
      return openServerTransportEncrypted(port);
    } else {
      return (new TServerSocket(port));
    }
  }

  public TServerTransport openServerTransportEncrypted(int port)
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

  /*
   * open Binary *********************
   */
  public TTransport openClientTransport(String server_host, int port)
          throws org.apache.thrift.TException {
    if (transportType == TransportType.encryptedClientSpecifiedTrustStore) {
      return openClientTransportEncrypted(server_host, port);
    } else {
      return (new TSocket(server_host, port));
    }
  }

  public TTransport openClientTransportEncrypted(String server_host, int port)
          throws org.apache.thrift.TException {
    // Used to set Socket.setSoTimeout ms. 0 == inifinite.
    int socket_so_timeout_ms = 0;
    if (transportType == TransportType.encryptedClientPermissive) {
      return TSSLTransportFactory.getClientSocket(
              server_host, port, socket_so_timeout_ms);
    }
    TSSLTransportFactory.TSSLTransportParameters params =
            new TSSLTransportFactory.TSSLTransportParameters();
    params.setTrustStore(trust_store_name,
            (trust_store_password != null) ? new String(trust_store_password) : null);
    params.requireClientAuth(false);

    return TSSLTransportFactory.getClientSocket(
            server_host, port, socket_so_timeout_ms, params);
  }

  private TrustManager[] trustManagers;
  // private boolean trust_all_certs = false;

  // TODO MAT the latest set of changes here appear to use a
  // SocketTransportProperties as the deciding mechanism of what
  // kind of connection to create encrypted or unencrypted
  // but the decision of which entry point to call has been passed
  // upstream to the caller
  // I have introduced this hack currently to get around broken
  // catalog when certs are enabled to allow the call to open
  // a connection to know to use and encrypted one when requested
  // to from upstream (ie a trust store was explicitly give in the
  // case of catalog connection)
  private TransportType transportType = null;
  private String trust_store_name = null;
  private char[] trust_store_password = null;
  private String key_store_name = null;
  private char[] key_store_password = null;
}
