package com.mapd.common;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.thrift.TException;
import org.apache.thrift.transport.THttpClient;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.LoggerFactory;
import javax.net.ssl.SSLContext;
import org.apache.http.ssl.SSLContexts;
import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import javax.net.ssl.X509TrustManager;
import java.io.File;
import java.io.FileInputStream;
import java.security.KeyStore;
import java.security.cert.X509Certificate;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import sun.net.www.http.HttpClient;

import static javax.print.attribute.standard.ReferenceUriSchemesSupported.HTTPS;

public class SockTransportProperties {
  final static org.slf4j.Logger logger =
          LoggerFactory.getLogger(SockTransportProperties.class);

  public SockTransportProperties(String ts_name, String ts_passwd) throws Exception {
    trust_store_name = ts_name;
    trust_store_password = ts_passwd;
    if (trust_store_name != null && !trust_store_name.isEmpty()) {
      java.io.FileInputStream fis = null;
      try {
        fis = new java.io.FileInputStream(trust_store_name);
        KeyStore ks = KeyStore.getInstance(KeyStore.getDefaultType());
        ks.load(fis, trust_store_password.toCharArray());
      } catch (Exception eX) {
        String err_str = new String("Trust store [" + trust_store_name + "] Error");
        logger.error(err_str, eX);
        throw(eX);
      }
    }
  }

  static public TTransport openHttpsClientTransport(
          String server_host, int port, SockTransportProperties sockManager)
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
  if (sockManager != null) {
    File trust_store = new File(sockManager.trust_store_name);
    sc = SSLContexts.custom()
                 .loadTrustMaterial(
                         trust_store, sockManager.trust_store_password.toCharArray())
                 .build();
  } else {
    sc = SSLContexts.custom().useProtocol("SSL").build();
    sc.init(null, trustAllCerts, new java.security.SecureRandom());
  }

  SSLConnectionSocketFactory factory = new SSLConnectionSocketFactory(
          sc, SSLConnectionSocketFactory.BROWSER_COMPATIBLE_HOSTNAME_VERIFIER);

  CloseableHttpClient cHC = HttpClients.custom().setSSLSocketFactory(factory).build();
  transport = new THttpClient("https://" + server_host + ":" + port, cHC);
} catch (Exception ex) {
  String err_str = new String("Exception:" + ex.getClass().getCanonicalName()
          + " thown. Unable to create Secure socket for the HTTPS connection");
  logger.error(err_str, ex);
  throw ex;
}
return transport;
}

static public TTransport openHttpClientTransport(
        String server_host, int port, SockTransportProperties sockManager)
        throws org.apache.thrift.TException {
  String url = "http://" + server_host + ":" + port;
  return (new THttpClient(url));
}

static public TTransport openClientTransport(
        String server_host, int port, SockTransportProperties sockManager)
        throws org.apache.thrift.TException {
  if (sockManager == null) {
    return (new TSocket(server_host, port));
  }
  TSSLTransportFactory.TSSLTransportParameters params =
          new TSSLTransportFactory.TSSLTransportParameters();
  params.setTrustStore(sockManager.trust_store_name, sockManager.trust_store_password);
  params.requireClientAuth(false);

  return TSSLTransportFactory.getClientSocket(server_host, port, 10000, params);
}

public String trust_store_name;
public String trust_store_password;
}
