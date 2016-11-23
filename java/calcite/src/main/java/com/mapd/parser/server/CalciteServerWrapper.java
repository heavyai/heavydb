/*
 *  Some cool MapD License
 */
package com.mapd.parser.server;

import org.apache.thrift.server.TServer;
import org.apache.thrift.server.TThreadPoolServer;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;
import com.mapd.thrift.calciteserver.CalciteServer.Processor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class CalciteServerWrapper implements Runnable {
  private final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteServerWrapper.class);
  private final CalciteServerHandler handler;
  private final Processor processor;
  private TServer server;
  private int mapDPort = 9091;
  private String dataDir = ("data/");
  private int calcitePort = 9093;
  private boolean shutdown = false;

  public CalciteServerWrapper(){
    handler = new CalciteServerHandler(mapDPort, dataDir, null);
    processor = new com.mapd.thrift.calciteserver.CalciteServer.Processor(handler);
  }

  public CalciteServerWrapper(int calcitePort, int mapDPort, String dataDir, String extensionFunctionsAstFile){
    handler = new CalciteServerHandler(mapDPort, dataDir, extensionFunctionsAstFile);
    processor = new com.mapd.thrift.calciteserver.CalciteServer.Processor(handler);
    this.calcitePort = calcitePort;
    this.mapDPort = mapDPort;
  }

  private void startServer(com.mapd.thrift.calciteserver.CalciteServer.Processor processor) {
    try {
      TServerTransport serverTransport = new TServerSocket(calcitePort);
      server = new TThreadPoolServer(new TThreadPoolServer.Args(serverTransport).processor(processor));

      MAPDLOGGER.info("Starting a threaded pool server... Listening on port "+ calcitePort + " MapD on port "+
                      mapDPort);
      handler.setServer(server);
      server.serve();

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void stopServer(){
    server.stop();
    shutdown = true;
  }

  @Override
  public void run() {
    startServer(processor);
  }

  boolean shutdown() {
    return shutdown;
  }
}
