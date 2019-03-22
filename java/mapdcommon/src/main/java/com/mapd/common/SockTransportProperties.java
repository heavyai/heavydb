package com.mapd.common;

import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.thrift.transport.THttpClient;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.slf4j.LoggerFactory;

import javax.net.ssl.SSLContext;

import org.apache.http.ssl.SSLContexts;

import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import javax.net.ssl.X509TrustManager;
import java.security.KeyStore;

import java.security.cert.X509Certificate;

import org.apache.http.conn.ssl.SSLConnectionSocketFactory;

public class SockTransportProperties {
  final static org.slf4j.Logger logger =
          LoggerFactory.getLogger(SockTransportProperties.class);

  public SockTransportProperties(String ts_name, String ts_passwd) throws Exception {
    // This constructor will always try and load trust data.  Either from the specified
    // trust store file (ts_name) or from the java default trust stores.
    if (ts_passwd != null && !ts_passwd.isEmpty())
      trust_store_password = ts_passwd.toCharArray();

    if (ts_name != null && !ts_name.isEmpty()) trust_store_name = ts_name;

    KeyStore kS = null;
    java.io.FileInputStream fis = null;
    try {
      kS = KeyStore.getInstance(KeyStore.getDefaultType());
      if (ts_name != null && !ts_name.isEmpty())
        fis = new java.io.FileInputStream(ts_name);
      // If using the default key store then fis should be null
      kS.load(fis, trust_store_password);
    } catch (Exception eX) {
      String err_str = new String("Trust store [" + ts_name + "] Error");
      logger.error(err_str, eX);
      throw(eX);
    }
    initializeAcceptedIssuers(kS);
  }

  public SockTransportProperties(boolean load_trust_store) throws Exception {
    // This constructor will either not bother loading trust data (and then trust all
    // server certs or load from the java default trust stores.
    trust_all_certs = !load_trust_store;
    if (load_trust_store) initializeAcceptedIssuers((KeyStore) null);
  }

  private void initializeAcceptedIssuers(KeyStore kS) throws Exception {
    TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance("PKIX");
    trustManagerFactory.init(kS);
    trustManagers = trustManagerFactory.getTrustManagers();
    X509TrustManager x509TrustManager = (X509TrustManager) trustManagers[0];
  }

  /*
   * open HTTPS  *********************
   */
  public TTransport openHttpsClientTransport(String server_host, int port)
          throws Exception {
    // Simple TrustManager to trust all certs
    TrustManager[] trustAllCerts = new TrustManager[] {new X509TrustManager(){
            public java.security.cert.X509Certificate[] getAcceptedIssuers(){return null;
  }

  public void checkClientTrusted(X509Certificate[] certs, String authType) {}

  public void checkServerTrusted(X509Certificate[] certs, String authType) {}
}
}
;

TTransport transport = null;
try {
  // Build a regular apache ClosableHttpClient based on a SSL connection
  // that can be passed to the apache thrift THttpClient constructor.

  SSLContext sc = null;
  sc = SSLContexts.custom().useProtocol("SSL").build();
  SSLConnectionSocketFactory sslConnectionSocketFactory = null;
  if (!trust_all_certs) {
    sc.init(null, trustManagers, new java.security.SecureRandom());
    sslConnectionSocketFactory = new SSLConnectionSocketFactory(
            sc, SSLConnectionSocketFactory.BROWSER_COMPATIBLE_HOSTNAME_VERIFIER);
  } else {
    sc.init(null, trustAllCerts, new java.security.SecureRandom());
    sslConnectionSocketFactory = new SSLConnectionSocketFactory(
            sc, SSLConnectionSocketFactory.ALLOW_ALL_HOSTNAME_VERIFIER);
  }
  CloseableHttpClient closeableHttpClient =
          HttpClients.custom().setSSLSocketFactory(sslConnectionSocketFactory).build();
  transport = new THttpClient("https://" + server_host + ":" + port, closeableHttpClient);
} catch (Exception ex) {
  String err_str = new String("Exception:" + ex.getClass().getCanonicalName()
          + " thown. Unable to create Secure socket for the HTTPS connection");
  logger.error(err_str, ex);
  throw ex;
}

return transport;
}

/*
 * opern HTTP *********************
 */
public TTransport openHttpClientTransport(String server_host, int port)
        throws org.apache.thrift.TException {
  String url = "http://" + server_host + ":" + port;
  return (new THttpClient(url));
}

/*
 * opern Binary *********************
 */
public TTransport openClientTransport(String server_host, int port) {
  return (new TSocket(server_host, port));
}

public TTransport openClientTransport_encryted(String server_host, int port)
        throws org.apache.thrift.TException {
  // Used to set Socket.setSoTimeout ms. 0 == inifinite.
  int socket_so_timeout_ms = 0;
  if (trust_all_certs)
    return TSSLTransportFactory.getClientSocket(server_host, port, socket_so_timeout_ms);

  TSSLTransportFactory.TSSLTransportParameters params =
          new TSSLTransportFactory.TSSLTransportParameters();
  String p = trust_store_password.toString();
  params.setTrustStore(trust_store_name,
          (trust_store_password != null) ? new String(trust_store_password) : null);
  params.requireClientAuth(false);

  return TSSLTransportFactory.getClientSocket(
          server_host, port, socket_so_timeout_ms, params);
}

private TrustManager[] trustManagers;
private boolean trust_all_certs;
private String trust_store_name = null;
private char[] trust_store_password = null;
}
